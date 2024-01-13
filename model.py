import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import logging
import torchstat
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

from registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=1)
    def forward(self, rgb, other):
        x, y = self.conv1(rgb), self.conv2(other)
        x = torch.cat([x, y], dim=1)
        return F.relu(x)

@MODEL_REGISTRY.register()
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans=4, expan_att_chans=4):
        super(ChannelAttention, self).__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.t2 = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv_text = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 2, 1, groups=embed_dim),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 2),
        )
        self.group_qkv_rgb = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim),
            Rearrange('B (C E)  H W -> B E C H W', E=expan_att_chans * 3),
        )
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def attnfun(self, q, k, v, t):
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * t
        return attn.softmax(dim=-1) @ v

    def forward(self, x, text):
        B, C, H, W = x.size()
        qRGB, kRGB, vRGB = self.group_qkv_rgb(x).contiguous().chunk(3, dim=1)
        kT, vT = self.group_qkv_text(text).contiguous().chunk(2, dim=1)
        C_exp = self.expan_att_chans * C

        qRGB = qRGB.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        kRGB, kT = kRGB.view(B, self.num_heads, C_exp // self.num_heads, H * W), kT.view(B, self.num_heads,
                                                                                         C_exp // self.num_heads, H * W)
        vRGB, vT = vRGB.view(B, self.num_heads, C_exp // self.num_heads, H * W), vT.view(B, self.num_heads,
                                                                                         C_exp // self.num_heads,
                                                                                         H * W),
        x_ = self.attnfun(qRGB, kRGB, vRGB, self.t)
        x_ = self.attnfun(x_, kT, vT, self.t2)
        x_ = rearrange(x_, "B C X (H W)-> B (X C) H W", B=B, W=W, H=H, C=self.num_heads).contiguous()
        x_ = self.group_fus(x_)
        return x_
class TCF(nn.Module):
    def __init__(self, embed_dim, squeezes, shuffle, expan_att_chans):
        super(TCF, self).__init__()
        self.embed_dim = embed_dim  # 16
        sque_ch_dim = embed_dim // squeezes[0]  # 16/4
        shuf_sp_dim = int(sque_ch_dim * (shuffle ** 2))  # 4*16**2=1024
        sque_sp_dim = shuf_sp_dim // squeezes[1]  # 128

        self.sque_ch_dim = sque_ch_dim  # 4
        self.shuffle = shuffle  # 16
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        self.s2c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, sque_ch_dim, 1),
                nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim)
            ) for _ in range(2)
        ])

        self.g1 = ChannelAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.c2s = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1)
        )

    def forward(self, x, text):
        x = self.s2c[0](x)
        text = self.s2c[1](text)
        x = self.g1(x, text)
        x = self.c2s(x)
        return x


@MODEL_REGISTRY.register()
class SDO(nn.Module):
    def __init__(self, embed_dim):
        super(SDO, self).__init__()
        self.embed_dim = embed_dim
        self.pre_conv = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1) for _ in range(2)
        ])
        self.post_conv = nn.Conv2d(embed_dim * 2, embed_dim, 1)
    def forward(self, x, inf):
        x0, x1 = self.pre_conv[0](x), self.pre_conv[1](inf)
        x_ = F.gelu(x0) * torch.sigmoid(x1)
        x_ = self.post_conv(x_)
        return x_


@MODEL_REGISTRY.register()
class Identity(nn.Module):
    def __init__(self, dim, isLangue):
        super(Identity, self).__init__()
        self.dim = dim
        self.isLangue = isLangue
        self.identity = nn.Identity
    def forward(self, x):
        return self.identity(x)
class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 squeezes,
                 shuffle,
                 expan_att_chans,
                 ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.ModuleList([
            nn.Sequential(
                Rearrange('B C H W -> B (H W) C'),
                nn.LayerNorm(embed_dim)
            ) for _ in range(2)
        ])
        self.tcf = TCF(embed_dim, squeezes, shuffle, expan_att_chans)
        self.norm2 = nn.ModuleList([
            nn.Sequential(
                Rearrange('B C H W -> B (H W) C'),
                nn.LayerNorm(embed_dim)
            ) for _ in range(2)
        ])
        self.sdo = SDO(embed_dim)

    def forward(self, batch):
        x, text, inf = batch
        B, C, H, W = x.size()
        x_ = rearrange(self.norm1[0](x), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_text = rearrange(self.norm1[1](text), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_ = self.tcf(x_, x_text)
        x = x + x_

        x_ = rearrange(self.norm2[0](x), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x_inf = rearrange(self.norm2[1](inf), "B (H W) C -> B C H W", H=H, W=W).contiguous()
        x = x + self.sdo(x_, x_inf)
        return x, text, inf

class TextEncoder(nn.Module):
    def __init__(self, dim=16):
        super(TextEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim,3,1,1)
        )

    def forward(self, x,H, W):
        x = rearrange(x, "b (c w h)-> b c w h", w=16, h=16)
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
class TITFormer(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64, expan_att_chans=4,
                 refine_blocks=2, num_blocks=(4, 6, 6, 2), num_shuffles=(16, 8, 4, 2),
                 ch_sp_squeeze=[(4, 8), (4, 4), (4, 2), (4, 1)]):
        super(TITFormer, self).__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed_text = TextEncoder(embed_dim)
        self.patch_embed_inf = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.encoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))])

        self.decoder = nn.ModuleList([nn.Sequential(*[
            TransformerBlock(
                embed_dim * 2 ** i, ch_sp_squeeze[i],
                num_shuffles[i], expan_att_chans
            ) for _ in range(num_blocks[i])
        ]) for i in range(len(num_blocks))][::-1])

        self.downsampler = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(int(embed_dim * 2 ** i), int(embed_dim * 2 ** (i - 1)), 3, 1, 1),
                    nn.PixelUnshuffle(2)
                ) for i in range(len(num_blocks) - 1)
            ]).append(nn.Identity()) for _ in range(3)
        ])

        self.upsampler = nn.ModuleList([nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(int(embed_dim * 2 ** (i - 1)), int(embed_dim * 2 ** i), 3, 1, 1)
        ) for i in range(len(num_blocks) - 1)][::-1]).append(nn.Identity())
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                embed_dim, ch_sp_squeeze[0], num_shuffles[0], expan_att_chans
            ) for _ in range(refine_blocks)
        ])

        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x, text, inf):
        B, C, H, W = x.size()
        x_emb = self.patch_embed(x)
        x_emb_text = self.patch_embed_text(text, H, W)
        x_emb_inf = self.patch_embed_inf(inf)
        # Encoder
        x_ = x_emb
        x_text = x_emb_text
        x_inf = x_emb_inf
        x_ms = []

        for layer, sampler, sampler1, sampler2 in zip(
                self.encoder, self.downsampler[0], self.downsampler[1], self.downsampler[2]
        ):
            x_, _, _ = layer((x_, x_text, x_inf))
            x_ms.append((x_, x_text, x_inf))
            x_, x_text, x_inf = sampler(x_), sampler1(x_text), sampler2(x_inf)
        # Decoder
        x_ = 0
        x_ms.reverse()
        for (x_e, x_e_text, x_e_inf), layer, sampler in zip(
                x_ms, self.decoder, self.upsampler
        ):
            x_, _, _ = layer((x_ + x_e, x_e_text, x_e_inf))
            x_ = sampler(x_)

        # Refinement
        x_, _, _ = self.refinement((x_ + x_emb, x_emb_text, x_emb_inf))
        x_ = self.conv_last(x_) + x
        return x_
class Enhancer():
    def __init__(self, config):
        super(Enhancer, self).__init__()
        self.model = TITFormer( in_chans=config.channel, embed_dim=config.embed_dim,
                               expan_att_chans=config.expan_att_chans,
                               refine_blocks=2, num_blocks=config.num_blocks, num_shuffles=config.num_shuffles,
                               ch_sp_squeeze=config.ch_sp_squeeze, )
        self.model.to(config.device)
        self.config = config
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
    def train_on_batch(self, batch: list):
        inputs = batch["A_input"].to(self.config.device)
        labels = batch["A_exptC"].to(self.config.device)
        inf = batch["A_Inf"].to(self.config.device)
        text = batch["img_text"].to(self.config.device)
        outputs = self.model(inputs.detach(), text.detach(), inf.detach())
        loss = self.criterion(outputs, labels.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        psnr = kornia.metrics.psnr(
            ((outputs + 1) / 2).clamp(0, 1),
            ((labels.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        ssim = kornia.metrics.ssim(
            ((outputs + 1) / 2).clamp(0, 1),
            ((labels.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
        meter = {
            'loss': loss.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item()
        }
        return meter, (inputs, labels, outputs, inf, text)

    def sche_step(self):
        self.scheduler.step()

    def validate_on_batch(self, batch: list):
        inputs = batch["A_input"].to(self.config.device)
        labels = batch["A_exptC"].to(self.config.device)
        inf = batch["A_Inf"].to(self.config.device)
        text = batch["img_text"].to(self.config.device)

        with torch.no_grad():
            outputs = self.model(inputs.detach(), text.detach(), inf.detach())
        loss = self.criterion(outputs, labels.detach())

        psnr = kornia.metrics.psnr(
            ((outputs + 1) / 2).clamp(0, 1),
            ((labels.detach() + 1) / 2).clamp(0, 1),
            1,
        )
        ssim = kornia.metrics.ssim(
            ((outputs + 1) / 2).clamp(0, 1),
            ((labels.detach() + 1) / 2).clamp(0, 1),
            window_size=11,
        ).mean()
        meter = {
            'loss': loss.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item()
        }
        return meter, (inputs, labels, outputs, inf, text)



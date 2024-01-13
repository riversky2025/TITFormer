import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import os

suffixes = ['/*.png', '/*.jpg', '/*.bmp', '/*.tif']
from transformers import BertTokenizer

from registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageDataset_XYZ(Dataset):
    def __init__(self, option, mode="train", combined=True):
        self.mode = mode
        root = os.path.join(option.data_root, "fiveK")
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        self.set1_inf_files = list()
        self.set1_text_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set1_input_files[i][:-1] + ".png"))
            self.set1_text_files.append(os.path.join(root, "text", set1_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        self.set2_inf_files = list()
        self.set2_text_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set2_input_files[i][:-1] + ".png"))
            self.set2_text_files.append(os.path.join(root, "text", set2_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        self.test_inf_files = list()
        self.test_text_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", test_input_files[i][:-1] + ".png"))
            self.test_text_files.append(os.path.join(root, "text", test_input_files[i][:-1] + ".txt"))
        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files
            self.set1_inf_files = self.set1_inf_files + self.set2_inf_files
            self.set1_text_files = self.set1_text_files + self.set2_text_files

        self.encodings_dict = torch.load(os.path.join(root, 'encodings.pt'))

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], cv2.IMREAD_UNCHANGED)
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            img_inf = Image.open(self.set1_inf_files[index % len(self.set1_inf_files)])
        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], cv2.IMREAD_UNCHANGED)
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img_inf = Image.open(self.test_inf_files[index % len(self.test_inf_files)])
        W, H = img_exptC.size
        if W == 480:
            H = 720
        else:
            W = 720
        img_input = img_input.astype(np.float32) / 65535.0

        # 将图像从CIE XYZ颜色空间转换为sRGB颜色空间
        img_input = cv2.cvtColor(img_input, cv2.COLOR_XYZ2BGR)

        # 将图像从32位浮点数转换为8位整数
        img_input = (img_input * 255).clip(0, 255).astype(np.uint8)

        img_inf = img_inf.resize((W, H))
        img_input = cv2.resize(img_input, (W, H))
        img_exptC = img_exptC.resize((W, H))

        img_input = self.transform(img_input)[..., :H // 16 * 16, :W // 16 * 16]
        img_exptC = self.transform(img_exptC)[..., :H // 16 * 16, :W // 16 * 16]
        img_inf = self.transform(img_inf)[..., :H // 16 * 16, :W // 16 * 16]

        if self.mode == "train":
            if np.random.random() > 0.5:
                img_input = F.hflip(img_input)
                img_exptC = F.hflip(img_exptC)
                img_inf = F.hflip(img_inf)
            a = np.random.uniform(0.6, 1.4)
            img_input = F.adjust_brightness(img_input, a)

        filename, _ = os.path.splitext(img_name)
        return {"A_input": img_input, "A_exptC": img_exptC, "A_Inf": img_inf, "img_text": self.encodings_dict[filename],
                "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


@DATASET_REGISTRY.register()
class ImageDataset_sRGB(Dataset):
    def __init__(self, option, mode="train", combined=True):
        self.mode = mode
        root = os.path.join(option.data_root, "fiveK")

        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        self.set1_inf_files = list()
        self.set1_text_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "input", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set1_input_files[i][:-1] + ".png"))
            self.set1_text_files.append(os.path.join(root, "text", set1_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        self.set2_inf_files = list()
        self.set2_text_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root, "input", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set2_input_files[i][:-1] + ".png"))
            self.set2_text_files.append(os.path.join(root, "text", set2_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_inf_files = list()
        self.test_expert_files = list()
        self.test_text_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "input", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", test_input_files[i][:-1] + ".png"))
            self.test_text_files.append(os.path.join(root, "text", test_input_files[i][:-1] + ".txt"))
        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files
            self.set1_inf_files = self.set1_inf_files + self.set2_inf_files
            self.set1_text_files = self.set1_text_files + self.set2_text_files

        self.encodings_dict = torch.load(os.path.join(root, 'encodings.pt'))

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_inf = Image.open(self.set1_inf_files[index % len(self.set1_inf_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])


        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_inf = Image.open(self.test_inf_files[index % len(self.test_inf_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
        W, H = img_input.size
        if W == 480:
            H = 720
        else:
            W = 720
        img_inf = img_inf.resize((W, H))
        img_input = img_input.resize((W, H))
        img_exptC = img_exptC.resize((W, H))

        img_input = self.transform(img_input)[..., :H // 16 * 16, :W // 16 * 16]
        img_exptC = self.transform(img_exptC)[..., :H // 16 * 16, :W // 16 * 16]
        img_inf = self.transform(img_inf)[..., :H // 16 * 16, :W // 16 * 16]
        if self.mode == "train":
            if np.random.random() > 0.5:
                img_input = F.hflip(img_input)
                img_exptC = F.hflip(img_exptC)
                img_inf = F.hflip(img_inf)
            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_brightness(img_input, a)
            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_saturation(img_input, a)

        filename, _ = os.path.splitext(img_name)
        return {"A_input": img_input, "A_exptC": img_exptC, "A_Inf": img_inf, "img_text": self.encodings_dict[filename],
                "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


@DATASET_REGISTRY.register()
class PPR_ImageDataset_sRGB(Dataset):
    def __init__(self, option, mode="train"):
        self.mode = mode
        root = os.path.join(option.data_root, "ppr10K")
        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        self.set1_inf_files = list()
        self.set1_text_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "source", set1_input_files[i][:-1] + ".tif"))
            self.set1_expert_files.append(os.path.join(root, option.version, set1_input_files[i][:-1] + ".tif"))
            self.set1_inf_files.append(os.path.join(root, "Infrared", set1_input_files[i][:-1] + ".png"))
        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_inf_files = list()
        self.test_expert_files = list()
        self.test_text_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "source", test_input_files[i][:-1] + ".tif"))
            self.test_expert_files.append(os.path.join(root, option.version, test_input_files[i][:-1] + ".tif"))
            self.test_inf_files.append(os.path.join(root, "Infrared", test_input_files[i][:-1] + ".png"))
        self.encodings_dict = torch.load(os.path.join(root, 'encodings.pt'))
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_inf = Image.open(self.set1_inf_files[index % len(self.set1_inf_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_inf = Image.open(self.test_inf_files[index % len(self.test_inf_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img_inf = img_inf.resize(img_input.size)
        W, H = img_input.size
        if W == 540:
            H = 360
            W = 528
        else:
            H = 528
            W = 360
        img_inf = img_inf.resize((W, H))
        img_input = img_input.resize((W, H))
        img_exptC = img_exptC.resize((W, H))
        img_input = self.transform(img_input)[..., :H // 16 * 16, :W // 16 * 16]
        img_exptC = self.transform(img_exptC)[..., :H // 16 * 16, :W // 16 * 16]
        img_inf = self.transform(img_inf)[..., :H // 16 * 16, :W // 16 * 16]
        if self.mode == "train":

            if np.random.random() > 0.5:
                img_input = F.hflip(img_input)
                img_exptC = F.hflip(img_exptC)
                img_inf = F.hflip(img_inf)

            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_brightness(img_input, a)

            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_saturation(img_input, a)

        filename, _ = os.path.splitext(img_name)
        return {"A_input": img_input, "A_exptC": img_exptC, "A_Inf": img_inf, "img_text": self.encodings_dict[filename],
                "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)

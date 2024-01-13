function [] = fivek_calculate_metrics(source_dir, target_dir)

% Original Code: https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/master/average_psnr_ssim.m

path_list = dir(fullfile(target_dir,'*.png'));
img_num = length(path_list);
%calculate psnr
total_psnr = 0;
total_ssim = 0;
total_eab = 0;
if img_num > 0
   for j = 1:img_num
       image_name = path_list(j).name;
       gt = imread(fullfile(target_dir,image_name));
       % tif jpg
       input = imread(fullfile(source_dir,[image_name(1:end-3), 'png']));

       psnr_val = psnr(im2double(input), im2double(gt));
       total_psnr = total_psnr + psnr_val;

       ssim_val = ssim(input, gt);
       total_ssim = total_ssim + ssim_val;

       eab = sqrt(sum((rgb2lab(gt) - rgb2lab(input)).^2,3));
       eab = mean(eab(:));
       total_eab = total_eab + eab;
       fprintf('%d %f %f %f %s\n',j,psnr_val,ssim_val,eab,fullfile(source_dir,image_name));
   end
end
qm_psnr = total_psnr / img_num;
avg_ssim = total_ssim / img_num;
avg_color = total_eab / img_num;
fprintf('The avgerage psnr is: %f', qm_psnr);
fprintf('The avgerage ssim is: %f', avg_ssim);
fprintf('The avgerage lab is: %f', avg_color);

end


function [mssim, ssim_map] = ssim(img1, img2, K, window, L)
if (nargin < 2 || nargin > 5)
   mssim = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   mssim = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) || (N < 11))
	   mssim = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);	%
   K(1) = 0.01;					% default settings
   K(2) = 0.03;					%
   L = 255;                                     %
end

if (nargin == 3)
   if ((M < 11) || (N < 11))
	   mssim = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   mssim = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   mssim = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   mssim = -Inf;
	   ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   mssim = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   mssim = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   mssim = -Inf;
	   ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   mssim = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   mssim = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end


img1 = double(img1);
img2 = double(img2);
f = max(1,round(min(M,N)/256));
if(f>1)
    lpf = ones(f,f);
    lpf = lpf/sum(lpf(:));
    img1 = imfilter(img1,lpf,'symmetric','same');
    img2 = imfilter(img2,lpf,'symmetric','same');

    img1 = img1(1:f:end,1:f:end);
    img2 = img2(1:f:end,1:f:end);
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 && C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
	denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end


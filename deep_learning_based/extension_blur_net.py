import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset

class FixedSumOfGaussians(nn.Module):
    def __init__(self, sigma1, sigma2, ratio, downscale_factor, channels):
        super().__init__()

        # Adjust sigmas and ratio for downscaling
        sigma1 = sigma1 / downscale_factor
        sigma2 = sigma2 / downscale_factor
        ratio = ratio * (downscale_factor ** 2)

    # build 1D kernels on same support
        kernel_size = int(np.ceil(10 * max(sigma1, sigma2))) + 1
        radius = kernel_size // 2
        x = np.arange(-radius, radius+1)
        g1 = np.exp(-x**2/(2*sigma1**2))
        g2 = np.exp(-x**2/(2*sigma2**2))
        # raw sums
        S1 = g1.sum()**2
        S2 = g2.sum()**2
        C  = S1 + ratio * S2

        # make them tensors [C_out, C_in/groups, 1, K]
        g1h = torch.from_numpy(g1.astype(np.float32))[None,None,None,:].repeat(channels,1,1,1)
        g1v = g1h.permute(0,1,3,2)
        g2h = torch.from_numpy(g2.astype(np.float32))[None,None,None,:].repeat(channels,1,1,1)
        g2v = g2h.permute(0,1,3,2)

        # define depthwise convs
        self.g1h = nn.Conv2d(channels, channels, (1,2*radius+1),
                             groups=channels, bias=False, padding=(0,radius))
        self.g1v = nn.Conv2d(channels, channels, (2*radius+1,1),
                             groups=channels, bias=False, padding=(radius,0))
        self.g2h = nn.Conv2d(channels, channels, (1,2*radius+1),
                             groups=channels, bias=False, padding=(0,radius))
        self.g2v = nn.Conv2d(channels, channels, (2*radius+1,1),
                             groups=channels, bias=False, padding=(radius,0))

        # inject fixed weights and freeze
        with torch.no_grad():
            self.g1h.weight.copy_(g1h)
            self.g1v.weight.copy_(g1v)
            self.g2h.weight.copy_(g2h)
            self.g2v.weight.copy_(g2v)
        for p in self.parameters():
            p.requires_grad = False

        self.ratio = ratio
        self.C = float(C)

    def forward(self, x):
        # two separable blurs
        b1 = self.g1v(self.g1h(x))
        b2 = self.g2v(self.g2h(x))
        # mix & normalize exactly as your 2D kernel
        return (b1 + self.ratio * b2) / self.C


class UNetAdditive(nn.Module):
    def __init__(self, in_ch=4, base_ch=8, depth=5):
        super().__init__()
        # encoder: conv+ReLU blocks +
        #           downsample by 2× each time
        self.encs = nn.ModuleList()
        self.in_ch = in_ch
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.encs.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch*2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # decoder: upsample by 2×, concat skip, conv+ReLU
        self.decs = nn.ModuleList()
        for i in range(depth-1, -1, -1):
            in_ch = ch + base_ch*(2**i)
            out_ch = base_ch*(2**i)
            self.decs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
        # final 1×1 to predict Δm ≥ 0
        self.final = nn.Conv2d(base_ch, self.in_ch, 1)

    def forward(self, x):
        # x is in [0,1], shape [B,3,H,W]
        skips = []
        out = x
        # encoder
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
            out = F.avg_pool2d(out, 2)
        # bottleneck
        out = self.bottleneck(out)
        # decoder
        for dec in self.decs:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
            skip = skips.pop()
            out = torch.cat([out, skip], dim=1)
            out = dec(out)
        # predict positive Δm
        E = F.softplus(self.final(out))      # shape [B,C,H,W], ≥0
        return x + E                          # extended image

class ExtensionBlurNet(nn.Module):

    def __init__(self, in_ch=4, base_ch=8, depth=5,  sigma1=0.5, sigma2=40, ratio=0.000004375, downscale_factor=1):
        super().__init__()

        # UNet for additive blur
        self.unet = UNetAdditive(in_ch=in_ch, base_ch=base_ch, depth=depth)

        # fixed sum of Gaussians
        self.fixed_blur = FixedSumOfGaussians(sigma1, sigma2, ratio, downscale_factor, in_ch)


    def forward(self, x):
        # x is in [0,1], shape [B,C,H,W]
        blurred = self.fixed_blur(x)  # apply fixed blur
        bloomed = self.unet(blurred)  # predict Δm
        return bloomed  # shape [B,C,H,W]



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePairDataset(Dataset):
    def __init__(self, orig_imgs, filter_imgs, N=-1):
        assert orig_imgs.shape == filter_imgs.shape
        if N > 0:
            orig_imgs = orig_imgs[:N]
            filter_imgs = filter_imgs[:N]
        self.orig_imgs = orig_imgs
        self.filter_imgs = filter_imgs

    def __len__(self):
        return self.orig_imgs.shape[0]

    def crop(self, x, y, crop_size=512):
        C, H, W = x.shape
        i = np.random.randint(0, high=H-crop_size)
        j = np.random.randint(0, high=W-crop_size)
        return x[:, i:i+crop_size, j:j+crop_size], y[:, i:i+crop_size, j:j+crop_size]

    def __getitem__(self, idx):
        return self.crop(self.orig_imgs[idx], self.filter_imgs[idx])









































import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import Dataset



def gaussian_kernel(sigma, k_size):
    """
    Generate Gaussian kernels for each pixel based on sigma values.

    Args:
        sigma (torch.Tensor): Sigma values with shape (B, C, H, W)
        k_size (int): Kernel size (default: 3)

    Returns:
        torch.Tensor: Gaussian kernels with shape (B, C, H, W, k_size, k_size)
    """
    device = sigma.device
    B, C, H, W = sigma.shape
    radius = k_size // 2

    # Create kernel grid
    y, x = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32, device=device),
        torch.arange(-radius, radius + 1, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Reshape for broadcasting
    x = x.view(1, 1, 1, 1, k_size, k_size)  # (1, 1, 1, 1, k, k)
    y = y.view(1, 1, 1, 1, k_size, k_size)
    sigma = sigma.unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W, 1, 1)

    # Compute Gaussian weights
    coeff = 1.0 / (2 * torch.pi * sigma**2)
    exponent = -(x**2 + y**2) / (2 * sigma**2)
    weights = coeff * torch.exp(exponent)

    # Normalize kernels
    weights_sum = weights.sum(dim=(-2, -1), keepdim=True)
    weights = weights / weights_sum
    return weights


def adaptive_gaussian_conv2d(img, kernels, k_size):
    """
    Perform spatially adaptive Gaussian convolution.

    Args:
        img (torch.Tensor): Input image with shape (B, C, H, W)
        k_size (int): Kernel size (default: 3)

    Returns:
        torch.Tensor: Convolved output with shape (B, C, H, W)
    """
    # Generate Gaussian kernels
    B, C, H, W = img.shape

    # Unfold input image into patches
    pad = k_size // 2
    unfolded = F.unfold(img, kernel_size=k_size, padding=pad)  # (B, C*k*k, H*W)
    unfolded = unfolded.view(B, C, k_size*k_size, H, W)      # (B, C, k*k, H, W)
    unfolded = unfolded.permute(0, 1, 3, 4, 2)               # (B, C, H, W, k*k)

    # Reshape kernels and multiply with patches
    kernels_flat = kernels.view(B, C, H, W, -1)              # (B, C, H, W, k*k)
    output = (unfolded * kernels_flat).sum(dim=-1)           # (B, C, H, W)

    return output


class GaussianNet(nn.Module):
    def __init__(self, k_size):
        super(GaussianNet, self).__init__()
        self.k_size = k_size

        self.enc1 = nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1)   # -> (16, 256, 256)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 128, 128)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 64, 64)
        self.enc4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # -> (128, 32, 32)

        self.encoder_sigma = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 32 → 64
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 64 → 128
        self.dec3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)   # 128 → 256
        self.dec4 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)    # 256 → 512
        

    def forward(self, img):
        e1 = F.relu(self.enc1(img))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        x_encoded = F.relu(self.enc4(e3)) + 1e-3

        sigma = self.encoder_sigma(img) + 1e-3

        kernels = gaussian_kernel(sigma, self.k_size)
        diffused = adaptive_gaussian_conv2d(x_encoded, kernels, self.k_size)

        x = F.relu(self.dec1(diffused))
        x = F.relu(self.dec2(x + e3))
        x = F.relu(self.dec3(x + e2))
        x = self.dec4(x + e1)
        out = F.sigmoid(x + img)
        return out


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePairDataset(Dataset):
    def __init__(self, orig_imgs, filter_imgs):
        assert orig_imgs.shape == filter_imgs.shape
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
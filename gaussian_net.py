import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random



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
    def __init__(self, H, W, C, k_size):
        super(GaussianNet, self).__init__()
        self.k_size = k_size


        self.encoder_img = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 450x450
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 225x225
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 113x113
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 57x57
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 29x29
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 15x15
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 8x8
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU()
        )

        self.encoder_sigma = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 450x450
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 225x225
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 113x113
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 57x57
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 29x29
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 15x15
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# 8x8
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU()
        )

        # Decoder: 32x32x128 -> 900x900x3
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),   # 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 512x512
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 1024x1024
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),                        # Still 1024x1024
            nn.Tanh(),  # Output in range [-1, 1]
            nn.AdaptiveAvgPool2d((900, 900))  # Resize to exact 900x900
        )

    def forward(self, img):
        sigma = self.encoder_sigma(img) + 1e-10
        encoded_img = self.encoder_img(img)

        kernels = gaussian_kernel(sigma, self.k_size).to(img.device)
        diffused_img = adaptive_gaussian_conv2d(encoded_img, kernels, self.k_size)
        
        return self.decoder(diffused_img)


def set_seed(seed=42):
    torch.manual_seed(seed)              # Set seed for CPU
    torch.cuda.manual_seed(seed)         # Set seed for current GPU
    torch.cuda.manual_seed_all(seed)     # Set seed for all GPUs (if using multi-GPU)
    np.random.seed(seed)                 # Set seed for NumPy
    random.seed(seed)                    # Set seed for Python’s random module

    # Ensures deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
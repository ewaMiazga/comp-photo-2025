import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader



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

        self.encoder_img = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 651, 1009)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 326, 505)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 163, 253)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# -> (128, 82, 127)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# -> (128, 41, 64)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# -> (128, 41, 64)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))                          # -> (128, 32, 32)
        )

        self.encoder_sigma = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 651, 1009)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 326, 505)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (128, 163, 253)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# -> (128, 82, 127)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# -> (128, 41, 64)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# -> (128, 41, 64)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))                          # -> (128, 32, 32)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # -> (128, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # -> (128, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 512, 512)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> (16, 1024, 1024)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1),    # -> (4, 2048, 2048)
            nn.ReLU(),
        )
        self.final_resize = nn.Upsample(size=(1301, 2018), mode='bilinear', align_corners=False)


    def forward(self, img):
        sigma = self.encoder_sigma(img) + 1e-10
        encoded_img = self.encoder_img(img)

        kernels = gaussian_kernel(sigma, self.k_size).to(img.device)
        diffused_img = adaptive_gaussian_conv2d(encoded_img, kernels, self.k_size)
        
        return self.final_resize(self.decoder(diffused_img))


def set_seed(seed=42):
    torch.manual_seed(seed)              # Set seed for CPU
    torch.cuda.manual_seed(seed)         # Set seed for current GPU
    torch.cuda.manual_seed_all(seed)     # Set seed for all GPUs (if using multi-GPU)
    np.random.seed(seed)                 # Set seed for NumPy
    random.seed(seed)                    # Set seed for Python’s random module

    # Ensures deterministic behavior (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePairDataset(Dataset):
    def __init__(self, orig_imgs, filter_imgs):
        assert orig_imgs.shape == filter_imgs.shape
        self.orig_imgs = orig_imgs
        self.filter_imgs = filter_imgs

    def __len__(self):
        return self.orig_imgs.shape[0]

    def __getitem__(self, idx):
        return self.orig_imgs[idx], self.filter_imgs[idx]
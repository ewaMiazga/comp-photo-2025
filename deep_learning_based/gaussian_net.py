import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
import math
from torch.nn.functional import avg_pool2d



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


# class GaussianNet(nn.Module):
#     def __init__(self, k_size=3):
#         super(GaussianNet, self).__init__()
#         self.k_size = k_size

#         self.enc1 = nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1)   # -> (16, 256, 256)
#         self.enc2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # -> (32, 128, 128)
#         self.enc3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 64, 64)
#         self.enc4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # -> (128, 32, 32)

#         self.encoder_sigma = nn.Sequential(
#             nn.Conv2d(4+1, 16, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),
#             nn.ReLU()
#         )

#         self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(128, 64, 3, 1, 1))
#         self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(64, 32, 3, 1, 1))   # 64 → 128
#         self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(32, 16, 3, 1, 1))   # 128 → 256
#         self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(16, 4, 3, 1, 1))    # 256 → 512
        
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
                    
#     def forward(self, img):
#         e1 = F.relu(self.enc1(img))
#         e2 = F.relu(self.enc2(e1))
#         e3 = F.relu(self.enc3(e2))
#         x_encoded = F.relu(self.enc4(e3))
#         B, C, H, W = x_encoded.shape

#         luminance = img.mean(dim=1, keepdim=True)
#         sigma_input = torch.cat([img, luminance], dim=1)

#         sigma = self.encoder_sigma(sigma_input) # + 1e-3
#         sigma = F.softplus(sigma)
#         sigma = sigma.expand(B, C, H, W)

#         kernels = gaussian_kernel(sigma, self.k_size)
#         diffused = adaptive_gaussian_conv2d(x_encoded, kernels, self.k_size)

#         diffused = diffused + x_encoded

#         diffused = x_encoded
        

#         x = F.relu(self.dec1(diffused))
#         x = F.relu(self.dec2(x + e3))
#         x = F.relu(self.dec3(x + e2))
#         x = self.dec4(x + e1)
#         out = x + img
#         return out





def lorentzian_kernel_1d(gamma, k_size, dim):
    """
    Generate 1D Lorentzian (Cauchy) kernels for each pixel based on gamma values.

    Args:
        gamma (torch.Tensor): Gamma values with shape (B, C, H, W)
        k_size (int): Kernel size (odd)
        dim (str): 'h' for horizontal or 'v' for vertical (not used in calculation, for clarity)

    Returns:
        torch.Tensor: 1D Lorentzian kernels with shape (B, C, H, W, k_size)
    """
    device = gamma.device
    B, C, H, W = gamma.shape
    radius = k_size // 2

    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)  # (k_size,)
    coords = coords.view(1, 1, 1, 1, -1)  # (1, 1, 1, 1, k_size) for broadcasting

    gamma = gamma.unsqueeze(-1)  # (B, C, H, W, 1)

    # Lorentzian weight: 1 / (1 + (x/gamma)^2)
    weights = 1.0 / (1.0 + (coords / gamma) ** 2)

    # Normalize
    weights_sum = weights.sum(dim=-1, keepdim=True)
    weights = weights / (weights_sum + 1e-6)

    return weights


def gaussian_kernel_1d(sigma, k_size, dim):
    """
    Generate 1D Gaussian kernels for each pixel based on sigma values.

    Args:
        sigma (torch.Tensor): Sigma values with shape (B, C, H, W)
        k_size (int): Kernel size
        dim (str): 'h' for horizontal kernel, 'v' for vertical kernel (not strictly used in calculation, just for clarity)

    Returns:
        torch.Tensor: 1D Gaussian kernels with shape
                      (B, C, H, W, k_size)
    """
    device = sigma.device
    B, C, H, W = sigma.shape
    radius = k_size // 2

    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device) # (k_size,)

    # Reshape sigma for broadcasting with coordinates
    sigma = sigma.unsqueeze(-1) # (B, C, H, W, 1)

    # Compute Gaussian weights
    # For 1D, the normalization constant is 1 / (sqrt(2 * pi) * sigma)
    
    # --- FIX START ---
    # Convert 2 * torch.pi to a tensor before taking the square root
    pi_tensor = torch.tensor(2 * torch.pi, dtype=torch.float32, device=device)
    coeff = 1.0 / (torch.sqrt(pi_tensor) * sigma)
    # --- FIX END ---
    
    exponent = -(coords**2) / (2 * sigma**2)
    weights = coeff * torch.exp(exponent)

    # Normalize kernels
    weights_sum = weights.sum(dim=-1, keepdim=True)
    weights = weights / (weights_sum + 1e-6) # Add epsilon to prevent division by zero

    return weights


def adaptive_gaussian_conv2d_separable(img, sigma, k_size):
    """
    Perform spatially adaptive separable Gaussian convolution.

    Args:
        img (torch.Tensor): Input image with shape (B, C, H, W)
        sigma (torch.Tensor): Sigma values from the network (B, C, H, W)
        k_size (int): Kernel size

    Returns:
        torch.Tensor: Convolved output with shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    pad = k_size // 2

    # --- Horizontal Pass ---
    # Generate 1D horizontal kernels
    kernels_h = gaussian_kernel_1d(sigma, k_size, dim='h') # (B, C, H, W, k_size)

    # Unfold input image into horizontal patches
    # kernel_size=(1, k_size) means 1 row, k_size columns
    unfolded_h = F.unfold(img, kernel_size=(1, k_size), padding=(0, pad)) # (B, C*1*k, H*W)
    # Reshape and permute to align with kernels_h for element-wise multiplication
    unfolded_h = unfolded_h.view(B, C, k_size, H, W).permute(0, 1, 3, 4, 2) # (B, C, H, W, k_size)

    # Apply horizontal kernels
    img_h_blurred = (unfolded_h * kernels_h).sum(dim=-1) # (B, C, H, W)

    # --- Vertical Pass ---
    # Generate 1D vertical kernels
    kernels_v = gaussian_kernel_1d(sigma, k_size, dim='v') # (B, C, H, W, k_size)

    # Unfold horizontally blurred image into vertical patches
    # kernel_size=(k_size, 1) means k_size rows, 1 column
    unfolded_v = F.unfold(img_h_blurred, kernel_size=(k_size, 1), padding=(pad, 0)) # (B, C*k*1, H*W)
    # Reshape and permute to align with kernels_v for element-wise multiplication
    unfolded_v = unfolded_v.view(B, C, k_size, H, W).permute(0, 1, 3, 4, 2) # (B, C, H, W, k_size)

    # Apply vertical kernels
    output = (unfolded_v * kernels_v).sum(dim=-1) # (B, C, H, W)

    return output


def adaptive_lorentzian_conv2d_separable(img, sigma, k_size):
    """
    Perform spatially adaptive separable Gaussian convolution.

    Args:
        img (torch.Tensor): Input image with shape (B, C, H, W)
        sigma (torch.Tensor): Sigma values from the network (B, C, H, W)
        k_size (int): Kernel size

    Returns:
        torch.Tensor: Convolved output with shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    pad = k_size // 2

    # --- Horizontal Pass ---
    # Generate 1D horizontal kernels
    kernels_h = lorentzian_kernel_1d(gamma=sigma, k_size=k_size, dim='h') # (B, C, H, W, k_size)

    # Unfold input image into horizontal patches
    # kernel_size=(1, k_size) means 1 row, k_size columns
    unfolded_h = F.unfold(img, kernel_size=(1, k_size), padding=(0, pad)) # (B, C*1*k, H*W)
    # Reshape and permute to align with kernels_h for element-wise multiplication
    unfolded_h = unfolded_h.view(B, C, k_size, H, W).permute(0, 1, 3, 4, 2) # (B, C, H, W, k_size)

    # Apply horizontal kernels
    img_h_blurred = (unfolded_h * kernels_h).sum(dim=-1) # (B, C, H, W)

    # --- Vertical Pass ---
    # Generate 1D vertical kernels
    kernels_v = lorentzian_kernel_1d(gamma=sigma, k_size=k_size, dim='v') # (B, C, H, W, k_size)

    # Unfold horizontally blurred image into vertical patches
    # kernel_size=(k_size, 1) means k_size rows, 1 column
    unfolded_v = F.unfold(img_h_blurred, kernel_size=(k_size, 1), padding=(pad, 0)) # (B, C*k*1, H*W)
    # Reshape and permute to align with kernels_v for element-wise multiplication
    unfolded_v = unfolded_v.view(B, C, k_size, H, W).permute(0, 1, 3, 4, 2) # (B, C, H, W, k_size)

    # Apply vertical kernels
    output = (unfolded_v * kernels_v).sum(dim=-1) # (B, C, H, W)

    return output

class GaussianNet(nn.Module):
    def __init__(self, k_size=27):
        super(GaussianNet, self).__init__()
        self.k_size = k_size
        inchannel=4
        block_size=2
        
        self.block_size = block_size

        # image encoder
        self.conv1_1 = nn.Conv2d(inchannel, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        # decoder
        self.up6 = nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        out_channel = inchannel
        
        
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        

        # self.conv10 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        img = x
        B, C, H, W = img.shape
        # image encoder
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)

        # decoder
        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)

        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)

        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)

        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        # x = self.lrelu(x)

        # x = self.conv10(x)

        sigma = F.softplus(x) * 10 + 1e-6
        sigma = sigma.expand(B, C, H, W)



        # === Apply Adaptive Gaussian Blur ===
        kernels = gaussian_kernel(sigma, self.k_size)
        # x = adaptive_gaussian_conv2d(img, kernels, self.k_size)

        # x = adaptive_lorentzian_conv2d_separable(img, sigma, self.k_size)
        x = adaptive_gaussian_conv2d_separable(img, sigma, self.k_size)
        
        
        return x, sigma



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

        x = x[:, i:i+crop_size, j:j+crop_size]; y = y[:, i:i+crop_size, j:j+crop_size]

        scale_factor = 0.5
        x = x.unsqueeze(0)  # (1, C, H, W)
        x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        x = x.squeeze(0)  # (C, H/2, W/2)

        y = y.unsqueeze(0)  # (1, C, H, W)
        y = F.interpolate(y, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        y = y.squeeze(0)  # (C, H/2, W/2)

        return x, y

    def __getitem__(self, idx):
        return self.crop(self.orig_imgs[idx], self.filter_imgs[idx])
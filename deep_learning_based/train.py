import torch
from torch import optim
from gaussian_net import ImagePairDataset, set_seed, GaussianNet
import numpy as np
from torch.utils.data import random_split, DataLoader
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF



def mask_l2_loss(network_output, gt, loss_mask):
    return ((network_output - gt) ** 2 * loss_mask).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()



# --- Local Variance Map Calculation ---
def compute_local_variance_map(image, kernel_size=5, stride=1, padding=None):
    """
    Computes a local variance map for an image.
    Uses 1x1 convolution with weights for mean to allow arbitrary kernel_size
    and then mean of squares.

    Args:
        image (torch.Tensor): Input image tensor (B, C, H, W).
        kernel_size (int): Size of the local window (e.g., 5 for a 5x5 window).
        stride (int): Stride for the sliding window.
        padding (int, optional): Padding for the convolution. If None, it defaults to kernel_size // 2.

    Returns:
        torch.Tensor: Local variance map (B, C, H_out, W_out).
    """
    if padding is None:
        padding = kernel_size // 2

    # Prepare a kernel of ones for mean computation
    mean_kernel = torch.ones(image.shape[1], 1, kernel_size, kernel_size,
                             device=image.device, dtype=image.dtype) / (kernel_size**2)

    # Calculate local mean (E[X]) using depthwise convolution
    # Group convolution acts like applying a separate filter per channel
    local_mean = F.conv2d(image, mean_kernel, groups=image.shape[1],
                          padding=padding, stride=stride)

    # Calculate local mean of squares (E[X^2])
    local_mean_sq = F.conv2d(image**2, mean_kernel, groups=image.shape[1],
                             padding=padding, stride=stride)

    # Compute variance map: E[X^2] - (E[X])^2
    # Ensure variance is non-negative due to potential floating point inaccuracies
    variance_map = local_mean_sq - local_mean**2

    return variance_map

# --- Hierarchical Variance Loss Class ---
class HierarchicalVarianceLoss(nn.Module):
    def __init__(self, scales=3, variance_kernel_size=5, downsample_method='bilinear', weights=None):
        """
        Initializes the Hierarchical Variance Loss module.

        Args:
            scales (int): Number of scales to compute variance maps.
                          e.g., 3 means original, 1/2, 1/4 resolution.
            variance_kernel_size (int): Kernel size for local variance computation at each scale.
            loss_fn (callable): The elementary loss function to apply at each scale (e.g., F.l1_loss, F.mse_loss).
            downsample_method (str): 'avg_pool' or 'bilinear'. How to downsample the images.
            weights (list or None): Optional list of weights for each scale. If None, uses equal weights.
                                    Should have length `scales`. More weight usually given to coarser scales
                                    or the original scale.
        """
        super(HierarchicalVarianceLoss, self).__init__()
        self.scales = scales
        self.variance_kernel_size = variance_kernel_size
        self.loss_fn = nn.L1Loss(reduction='none')
        self.downsample_method = downsample_method

        if weights is None:
            self.weights = [1.0] * scales # Equal weights by default
        else:
            if len(weights) != scales:
                raise ValueError(f"Length of 'weights' must match 'scales' ({scales}).")
            self.weights = weights

    def forward(self, pred_image, target_image, mask=None):
        total_loss = 0.0

        for i in range(self.scales):
            # Downsample images for the current scale
            if i > 0:
                if self.downsample_method == 'avg_pool':
                    pred_image = F.avg_pool2d(pred_image, kernel_size=2, stride=2)
                    target_image = F.avg_pool2d(target_image, kernel_size=2, stride=2)
                    if mask is not None:
                        mask = F.avg_pool2d(mask, kernel_size=2, stride=2)
                elif self.downsample_method == 'bilinear':
                    new_size = (pred_image.shape[2] // 2, pred_image.shape[3] // 2)
                    pred_image = F.interpolate(pred_image, size=new_size, mode='bilinear', align_corners=False)
                    target_image = F.interpolate(target_image, size=new_size, mode='bilinear', align_corners=False)
                    if mask is not None:
                        mask = F.interpolate(mask, size=new_size, mode='bilinear', align_corners=False)
                else:
                    raise ValueError(f"Unknown downsample_method: {self.downsample_method}")

            # Compute variance maps at the current scale
            pred_variance_map = compute_local_variance_map(pred_image, kernel_size=self.variance_kernel_size)
            gt_variance_map = compute_local_variance_map(target_image, kernel_size=self.variance_kernel_size)

            # Calculate loss for the current scale and add with its weight
            scale_loss = self.loss_fn(pred_variance_map, gt_variance_map)
            if mask is not None:
                scale_loss = (scale_loss * mask).sum() / (mask.sum() + 1e-6)
            else:
                scale_loss = scale_loss.mean()
            total_loss += self.weights[i] * scale_loss

        return total_loss.mean()



def masked_tv_loss(x, mask):
    """ x: (B, C, H, W), mask: (B, 1, H, W) """
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    tv_x = (dx * mask_x).sum()
    tv_y = (dy * mask_y).sum()

    norm = mask_x.sum() + mask_y.sum() + 1e-6  # avoid division by zero

    return (tv_x + tv_y) / norm


def tv_loss(x):
    """ x: (B, C, H, W) """
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    tv_x = (dx).sum()
    tv_y = (dy).sum()

    return (tv_x + tv_y)

def main():
    parser = argparse.ArgumentParser(description="Trains the Gaussian net")
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=300)
    args = parser.parse_args()

    device = "cuda"
    set_seed(42)

    print("Loading data")
    orig_imgs = torch.load('dataset_raw/long_exp_chunk0.pt', weights_only=True)
    filter_imgs = torch.load('dataset_raw/filter_long_exp_chunk0.pt', weights_only=True)

    print("Loaded data")


    dataset = ImagePairDataset(orig_imgs.permute(3, 0, 1, 2), filter_imgs.permute(3, 0, 1, 2))
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.cuda.empty_cache()
    net = GaussianNet(k_size=21).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses = []

    print("Training the model")

    criterion = HierarchicalVarianceLoss()
    # weight_sigma = 1e-4
    # weight_pel = 1e-3

    weight_sigma = 1e-3
    weight_dark = 1e-2
    weight_bright = 5e-1



    for epoch in range(args.epochs):
        epoch_losses = []
        for original_img, filter_img in train_loader:
            original_img = original_img.to(device)
            filter_img = filter_img.to(device)

            optimizer.zero_grad()
            blurred_img, sigma = net(original_img)
            # loss_mask = torch.abs(original_img - filter_img) + 1e-3
            # loss = mask_l2_loss(blurred_img, filter_img, loss_mask)



            brightness = original_img.mean(dim=1, keepdim=True)

            # compute loss on variance difference
            loss_var = criterion(blurred_img, filter_img, mask=None)

            # compute l2 loss between images
            loss_l2 = l2_loss(blurred_img, filter_img) * 0.01

            # enforce sigma to be spatially smooth
            log_sigma = torch.log(sigma + 1e-6) * weight_sigma
            loss_sigma = tv_loss(log_sigma) * weight_sigma
            loss = loss_var + loss_sigma + loss_l2

            # enforce small sigma values in regions without highlight
            mask = (brightness < 0.9).float()
            blurred_mask = TF.gaussian_blur(mask, kernel_size=7, sigma=5)
            
            # enforce large sigma values in highlight regions  
            loss_dark = (sigma * (blurred_mask)).mean() * weight_dark
            loss = loss + loss_dark


            loss_bright = ((1.0 / (sigma + 1e-6)) * (1- blurred_mask)).mean() * weight_bright
            loss = loss + loss_bright
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        losses.append(np.mean(epoch_losses))
        print(f"Epoch {epoch}, Loss variance: {loss_var.item()}, Loss L2: {loss_l2.item()}, Loss sigma: {loss_sigma.item()}, Loss dark: {loss_dark.item()}, Loss bright: {loss_bright.item()}")
        # print(f"Epoch {epoch}, Loss: {loss_var.item()}")

        if epoch % 5 == 0:
            model_weights_path = "net.pth"
            torch.save(net.state_dict(), model_weights_path)
            print(f"Saved model weights at {model_weights_path}")

    print("Finished Training")
    model_weights_path = "net.pth"
    torch.save(net.state_dict(), model_weights_path)
    print(f"Saved model weights at {model_weights_path}")

    l1 = 0.0
    for original_img, filter_img in test_loader:
        original_img = original_img.to(device)
        filter_img = filter_img.to(device)     
        blurred_img = net(original_img)
        l1 += l1_loss(filter_img, blurred_img) * original_img.shape[0]
    
    l1 /= len(test_dataset)

    print(f"Mean test L1: {l1}")

if __name__ == "__main__":
    main()

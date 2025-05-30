import torch
from torch import optim
from extension_blur_net import ImagePairDataset, set_seed, ExtensionBlurNet
import numpy as np
from torch.utils.data import random_split, DataLoader
import argparse
import torch.nn.functional as F
from tqdm import tqdm
#
# def mask_l2_loss(network_output, gt, loss_mask):
#     return ((network_output - gt) ** 2 * loss_mask).mean()
#
# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()
#

def main():
    parser = argparse.ArgumentParser(description="Trains the extension blur net")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    batch_size = args.batch_size
    print("Loading data")
    orig_imgs = torch.load('../dataset_raw/long_exp.pt', weights_only=True) # (C, H, W, N)
    filter_imgs = torch.load('../dataset_raw/filter_long_exp.pt', weights_only=True) # (C, H, W, N)

    print("Loaded data")
    dataset = ImagePairDataset(orig_imgs.permute(3, 0, 1, 2), filter_imgs.permute(3, 0, 1, 2), N=5)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2]) # Shape (C, H, W, N)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.cuda.empty_cache()
    net = ExtensionBlurNet(downscale_factor=2).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses = []

    print("Training the model")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        epoch_losses = []
        for original_img, filter_img in tqdm(train_loader): # (B,
            print("Original image shape:", original_img.shape)
            original_img = original_img.to(device)
            filter_img = filter_img.to(device)

            optimizer.zero_grad()
            bloomed_img = net(original_img)
            loss = F.mse_loss(bloomed_img, filter_img)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        losses.append(np.mean(epoch_losses))
        print(f"Epoch {epoch+1}, Loss: {losses[-1]}")

    print("Finished Training")
    model_weights_path = "extension_blur_net.pth"
    torch.save(net.state_dict(), model_weights_path)
    print(f"Saved model weights at {model_weights_path}")

    mse_test_loss = 0
    l1_test_loss = 0
    print("Testing the model")
    for original_img, filter_img in test_loader:
        original_img = original_img.to(device)
        filter_img = filter_img.to(device)     
        bloomed_img = net(original_img)
        mse_test_loss += F.mse_loss(bloomed_img, filter_img)*original_img.shape[0]
        l1_test_loss += torch.abs(bloomed_img - filter_img).mean()*original_img.shape[0]
    
    mse_test_loss /= len(test_dataset)
    l1_test_loss /= len(test_dataset)

    print(f"Mean test MSE: {mse_test_loss}")
    print(f"Mean test L1: {l1_test_loss}")

if __name__ == "__main__":
    main()

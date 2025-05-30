import torch
from torch import optim
from gaussian_net import ImagePairDataset, set_seed, GaussianNet
import numpy as np
from torch.utils.data import random_split, DataLoader
import argparse


def mask_l2_loss(network_output, gt, loss_mask):
    return ((network_output - gt) ** 2 * loss_mask).mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def main():
    parser = argparse.ArgumentParser(description="Trains the Gaussian net")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    args = parser.parse_args()

    device = "cuda"
    set_seed(42)

    print("Loading data")
    orig_imgs = torch.load('dataset_raw/long_exp.pt', weights_only=True)
    filter_imgs = torch.load('dataset_raw/filter_long_exp.pt', weights_only=True)

    print("Loaded data")

    dataset = ImagePairDataset(orig_imgs.permute(0, 3, 1, 2), filter_imgs.permute(0, 3, 1, 2))
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    torch.cuda.empty_cache()
    net = GaussianNet(k_size=17).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses = []

    print("Training the model")

    for epoch in range(args.epochs):
        epoch_losses = []
        for original_img, filter_img in train_loader:
            original_img = original_img.to(device)
            filter_img = filter_img.to(device)

            optimizer.zero_grad()
            blurred_img = net(original_img)
            loss_mask = torch.abs(original_img - filter_img) + 1e-3
            loss = mask_l2_loss(blurred_img, filter_img, loss_mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        losses.append(np.mean(epoch_losses))
        print(f"Epoch {epoch}, Loss: {losses[-1]}")

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

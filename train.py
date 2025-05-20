import torch
from torch import optim
from gaussian_net import ImagePairDataset, set_seed, GaussianNet
import numpy as np


def mask_l2_loss(network_output, gt, loss_mask):
    return ((network_output - gt) ** 2 * loss_mask).mean()


def main():
    device = "cuda"
    print("Loading data")
    orig_imgs = torch.load('dataset_raw/long_exp.pt', weights_only=True)
    filter_imgs = torch.load('dataset_raw/filter_long_exp.pt', weights_only=True)
    print("Loaded data")

    dataset = ImagePairDataset(orig_imgs.permute(0, 3, 1, 2), filter_imgs.permute(0, 3, 1, 2))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


    torch.cuda.empty_cache()
    set_seed(42)
    net = GaussianNet(k_size=17).to(device)

    num_epochs = 200
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    losses = []

    print("Training the model")

    for epoch in range(num_epochs):
        epoch_losses = []
        for original_img, filter_img in dataloader:
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


if __name__ == "__main__":
    main()

import torch
import os
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from models.vqvae import VQVAE
from dataset import STL10Dataset

def load_data(means, stds):

    training_transforms = transforms.Compose(
        [transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)]
    )

    train_dataset = STL10Dataset(root_dir = "data/stl10/train_images", transform = training_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 64,
        shuffle = True, 
        num_workers = 8,
        pin_memory=True
    )
    return train_dataloader

def visualize(imgs, reconstructions, means, stds, out_path):
    means = torch.tensor(means, device = imgs.device).view(1, 3, 1, 1)
    stds = torch.tensor(stds, device = imgs.device).view(1, 3, 1, 1)

    if not os.path.exists(out_path):
        os.makedir(out_path)
    
    comparison = torch.cat([imgs, reconstructions], dim=0)
    comparison = comparison * stds + means

    grid = make_grid(
        tensor=comparison, 
        nrow=64, 
        padding=2, 
        pad_value=1.0
    )

    save_image(
        grid, 
        fp = os.path.join(out_path, f"output.png")
    )

def evaluate(checkpoint_path, dataloader):
    model = VQVAE()
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model.to(device)

    model.load_checkpoints(checkpoint_path)

    model.eval()

    imgs = next(iter(dataloader))
    imgs = imgs.to(device)

    with torch.no_grad():
        reconstruction, _, _ = model(imgs)

    return imgs, reconstruction
    
def main():
    training_data_means = [0.4471, 0.4402, 0.4070]
    training_data_stds = [0.2553, 0.2515, 0.2665]

    dataloader = load_data(means = training_data_means, stds = training_data_stds)

    imgs, reconstructions = evaluate(checkpoint_path = "checkpoints/vqvae_epoch_4.pt", 
                                     dataloader = dataloader)

    visualize(imgs, reconstructions, training_data_means, training_data_stds, "results")

if __name__ == "__main__":
    main()
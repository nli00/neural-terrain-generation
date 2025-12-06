import torch
import os
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from models.vqvae import VQVAE
from dataset import STL10Dataset
import argparse
import yaml

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
        shuffle = False, 
        num_workers = 8,
        pin_memory=True
    )
    return train_dataloader

def visualize(imgs, reconstructions, means, stds):
    means = torch.tensor(means, device = imgs.device).view(1, 3, 1, 1)
    stds = torch.tensor(stds, device = imgs.device).view(1, 3, 1, 1)
    
    comparison = torch.cat([imgs, reconstructions], dim=0)
    comparison = comparison * stds + means # un normalize the images

    grid = make_grid(
        tensor=comparison, 
        nrow=64, 
        padding=2, 
        pad_value=1.0
    )

    return grid

def evaluate(checkpoint_path, dataloader, config):
    model = VQVAE(config)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, help = "path to directory with checkpoint", required = True)
    parser.add_argument("--checkpoint", type = str, required = True)
    args = parser.parse_args()

    with open(os.path.join("checkpoints", args.checkpoint_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    dataloader = load_data(means = config['means'], stds = config['stds'])

    imgs, reconstructions = evaluate(checkpoint_path = os.path.join("checkpoints", args.checkpoint_dir, args.checkpoint), 
                                     dataloader = dataloader, config = config)

    grid = visualize(imgs, reconstructions, config['means'], config['stds'])

    out_path = 'results'

    if not os.path.exists(out_path):
        os.makedir(out_path)

    # dirs = os.listdir(out_path)
    # if len(dirs) == 0:
    #     highest_index = 0
    # else:
    #     highest_index = int(sorted(dirs)[-1].split('_')[-1])

    # out_path = os.path.join(out_path, f"test_{highest_index + 1}")
    out_path = os.path.join(out_path, f"{args.checkpoint_dir}_{args.checkpoint[:-3]}")
    os.mkdir(out_path)

    # with open(os.path.join(out_path, "config.yaml"), 'w') as f:
    #     yaml.dump(config, f)

    save_image(
        grid, 
        fp = os.path.join(out_path, f"output.png")
    )

if __name__ == "__main__":
    main()
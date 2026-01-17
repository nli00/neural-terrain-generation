import torch
import os
from torchvision.utils import save_image, make_grid
from torchvision.transforms import v2
import numpy as np

from models.vqvae import VQVAE
from dataset import STL10Dataset, USGSDataset
import argparse
import yaml

def load_data(means, stds, dataset, resolution):

    # TODO: replace PIL pipeline fully with torch tensors for performance
    # So that means all this casing is temporary
    if dataset == 'STL10':
        training_transforms = v2.Compose(
            [v2.Resize((resolution, resolution)),
            v2.ToTensor(),
            v2.Normalize(mean=means, std=stds)
        ])
    elif dataset == 'USGS':
        training_transforms = v2.Compose([
            v2.Resize((resolution, resolution)),
            v2.Normalize(mean=means, std=stds)
        ])
    else:
        raise NotImplementedError

    if dataset == 'STL10':
        data = STL10Dataset(root_dir = "data/stl10/test_images", transform = training_transforms)
    elif dataset == 'USGS':
        data = USGSDataset(root_dir = "data/test_geo", transform = training_transforms)
    else:
        raise NotImplementedError
    
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size = 64,
        shuffle = False, 
        num_workers = 8,
        pin_memory=True
    )
    return dataloader

def visualize(imgs, reconstructions, means, stds):
    # Channels is at [1]
    if imgs.shape[1] == 3:
        means = torch.tensor(means, device = imgs.device).view(1, 3, 1, 1)
        stds = torch.tensor(stds, device = imgs.device).view(1, 3, 1, 1)
    elif imgs.shape[1] == 1:
        means = means[0]
        stds = stds[0]
    else:
        raise NotImplementedError
    
    comparison = torch.cat([imgs, reconstructions], dim=0)
    comparison = comparison * stds + means # un normalize the images

    grid = make_grid(
        tensor=comparison, 
        nrow=64, 
        padding=0, 
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

    dataloader = load_data(means = config['means'], stds = config['stds'], dataset = config['dataset'], resolution = config['resolution'])

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
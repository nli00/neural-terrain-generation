import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

from dataset import STL10Dataset


def main():
    num_epochs = 10

    training_data_means = [0.4471, 0.4402, 0.4070]
    training_data_stds = [0.2553, 0.2515, 0.2665]

    training_transforms = torch.nn.Sequential(
        [transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=training_data_means, std=[0.2553, 0.2515, 0.2665])]
    )

    train_dataset = STL10Dataset(root_dir = "data/stl10/train_images", transform = training_transforms)
    train_dataloader = torch.utils.data.Dataloader(
        train_dataset,
        batch_size = 64,
        shuffle = True, 
        num_workers = 8
    )

    for epoch in range(num_epochs):
        pass

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse

from dataset import STL10Dataset
from models.vqvae import VQVAE

class VQVAETrainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQVAE()
        self.model.to(self.device)
        self.reconstruction_loss_fn = nn.MSELoss()
        self.num_epochs = 10

        self.lr = .01
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr = self.lr
        )

        os.makedirs("results", exist_ok = True)
        os.makedirs("checkpoints", exist_ok = True)

    def load_data(self):
        training_data_means = [0.4471, 0.4402, 0.4070]
        training_data_stds = [0.2553, 0.2515, 0.2665]

        training_transforms = transforms.Compose(
            [transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=training_data_means, std=training_data_stds)]
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
    
    def train(self):
        train_dataloader = self.load_data()
        for epoch in range(self.num_epochs):
            self.model.train()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} : {self.num_epochs}")

            for imgs in pbar:
                imgs = imgs.to(self.device)
                decoded_images, _, vq_regularization_loss = self.model(imgs)

                reconstruction_loss = self.reconstruction_loss_fn(decoded_images, imgs)
                vq_loss = reconstruction_loss + vq_regularization_loss
                
                self.opt.zero_grad()
                vq_loss.backward()
                self.opt.step()

                # should save some images out around here

                pbar.set_postfix({
                    'VQ_Loss' : f"{vq_loss.item():.5f}",
                    'Recon_Loss' : f"{reconstruction_loss.item():.5f}",
                    'VQ_Reg_Loss' : f"{vq_regularization_loss.item():.5f}"
                })

            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"vqvae_epoch_{epoch}.pt"))

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    trainer = VQVAETrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
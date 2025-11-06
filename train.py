import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse
import yaml
import glob
from torch.utils.tensorboard import SummaryWriter
import shutil

from dataset import STL10Dataset
from models.vqvae import VQVAE

class VQVAETrainer:
    def __init__(self, config, out_dir, summary_writer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQVAE()
        self.model.to(self.device)
        self.reconstruction_loss_fn = nn.MSELoss()
        self.num_epochs = config['num_epochs']
        self.len_data = -1
        self.config = config
        self.writer = summary_writer
        self.resolution = config['resolution']
        self.out_dir = out_dir

        self.lr = .01
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr = self.lr
        )

        self.use_amp = config['use_amp']
        if self.use_amp:
            self.scaler = GradScaler()
        

        os.makedirs("results", exist_ok = True)
        os.makedirs("checkpoints", exist_ok = True)

        self.best_vq = float('inf')
        self.best_recon = float('inf')
        self.best_regularization = float('inf')

    def load_data(self):
        training_data_means = [0.4471, 0.4402, 0.4070]
        training_data_stds = [0.2553, 0.2515, 0.2665]

        training_transforms = transforms.Compose(
            [transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=training_data_means, std=training_data_stds)]
        )

        train_dataset = STL10Dataset(root_dir = self.config['train_dataset'], transform = training_transforms)
        self.len_data = len(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = 64,
            shuffle = True, 
            num_workers = 8,
            pin_memory = True
        )
        return train_dataloader
    
    def update_log(self, epoch, vq_loss, recon_loss, regularization_loss):
        self.writer.add_scalar("vq_loss", vq_loss, epoch)
        self.writer.add_scalar("recon_loss", recon_loss, epoch)
        self.writer.add_scalar("regularization_loss", regularization_loss, epoch)

        if recon_loss < self.best_recon:
             self.best_recon = recon_loss
        if regularization_loss < self.best_regularization:
             self.best_regularization = regularization_loss
        if vq_loss < self.best_vq:
            self.best_vq = vq_loss
            return True
        return False
    
    def cleanup_checkpoints(self, type):
        checkpoints = glob.glob(os.path.join(self.out_dir, f"*{type}*.pt"))
        for checkpoint in checkpoints:
            os.remove(checkpoint)
    
    def train(self):
        train_dataloader = self.load_data()
        for epoch in range(self.num_epochs):
            self.model.train()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} : {self.num_epochs}")
            epoch_vq_loss = 0
            epoch_reg_loss = 0
            epoch_recon_loss = 0

            for imgs in pbar:
                imgs = imgs.to(self.device)
                cur_batch_size = imgs.size(0)

                if self.use_amp:
                    with autocast(device_type=self.device):
                        decoded_images, _, vq_regularization_loss = self.model(imgs)

                        recon_loss = self.reconstruction_loss_fn(decoded_images, imgs)
                        vq_loss = recon_loss + vq_regularization_loss
                    
                    self.opt.zero_grad()
                    self.scaler.scale(vq_loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    decoded_images, _, vq_regularization_loss = self.model(imgs)

                    recon_loss = self.reconstruction_loss_fn(decoded_images, imgs)
                    vq_loss = recon_loss + vq_regularization_loss
                    
                    self.opt.zero_grad()
                    vq_loss.backward()
                    self.opt.step()

                # should save some images out around here

                pbar.set_postfix({
                    'VQ_Loss' : f"{vq_loss.item():.5f}",
                    'Recon_Loss' : f"{recon_loss.item():.5f}",
                    'VQ_Reg_Loss' : f"{vq_regularization_loss.item():.5f}"
                })

                epoch_vq_loss += vq_loss.item() * cur_batch_size
                epoch_reg_loss += vq_regularization_loss.item() * cur_batch_size
                epoch_recon_loss += recon_loss.item() * cur_batch_size
                self.writer.flush()

            epoch_vq_loss /= self.len_data
            epoch_recon_loss /= self.len_data
            epoch_reg_loss /= self.len_data

            pbar.set_postfix({
                    'VQ_Loss' : f"{epoch_vq_loss:.5f}",
                    'Recon_Loss' : f"{epoch_recon_loss:.5f}",
                    'VQ_Reg_Loss' : f"{epoch_reg_loss:.5f}"
                })
            
            if self.update_log(epoch_vq_loss, epoch_recon_loss, epoch_reg_loss, epoch):
                self.cleanup_checkpoints('best')
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"best_epoch_{epoch}.pt"))
            self.cleanup_checkpoints('latest')
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"latest_epoch_{epoch}.pt"))
            
            self.writer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, help = "name of config file in configs/", required = True)
    args = parser.parse_args()

    if not args.config.endswith(".yaml"):
        config_file = args.config + ".yaml"

    with open(os.path.join("configs", config_file)) as f:
        config = yaml.safe_load(f)
    
    out_dir = os.path.join("checkpoints", args.config)

    if os.path.exists(os.path.join("checkpoints", args.config)):
        print(f"Contining will overwrite existing checkpoints in checkpoints/{args.config}.")
        response = input("Type \"continue\" to continue.\n")
        if response != "continue":
            return
        # else:
        #     shutil.rmtree(os.path.join("checkpoints", args.config, 'logs'))
    else:
        os.mkdir(os.path.join("checkpoints", args.config))

    writer = SummaryWriter(os.path.join("checkpoints", args.config, 'logs'))
    
    with open(os.path.join("checkpoints", args.config, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    try:
        trainer = VQVAETrainer(config, out_dir, writer)
        trainer.train()
        writer.close()
    except KeyboardInterrupt:
        writer.close()

if __name__ == "__main__":
    main()
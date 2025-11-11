import torch
from torch.amp import autocast, GradScaler
from torchvision import transforms

import os
from tqdm import tqdm
import argparse
import lpips

import utils
from dataset import STL10Dataset
from models.vqvae import VQVAE

class VQVAETrainer:
    def __init__(self, config, out_dir, summary_writer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQVAE(config)
        self.model.to(self.device)

        # self.reconstruction_loss_fn = nn.MSELoss()
        self.reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')

        self.config = config
        self.num_epochs = config['num_epochs']
        self.start_epoch = 0
        self.out_dir = out_dir

        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr = self.config['lr']
        )

        # self.perceptual_loss_fn = lpips.LPIPS(net = 'vgg').to(device = self.device)

        self.use_amp = config['use_amp']
        if self.use_amp:
            self.scaler = GradScaler()

        os.makedirs("results", exist_ok = True)
        os.makedirs("checkpoints", exist_ok = True)

        self.logger = utils.Logger(summary_writer, self.out_dir)

    def load_data(self):
        training_data_means = self.config['means']
        training_data_stds = self.config['stds']

        training_transforms = transforms.Compose(
            [transforms.Resize((self.config['resolution'], self.config['resolution'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=training_data_means, std=training_data_stds)]
        )

        train_dataset = STL10Dataset(root_dir = self.config['train_dataset'], transform = training_transforms)
        self.len_data = len(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'],
            pin_memory = True
        )
        return train_dataloader
    
    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
    
    def train(self):
        train_dataloader = self.load_data()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} : {self.num_epochs}")
            losses = {'loss': 0,
                      'recon_loss': 0,
                      'reg_loss': 0}
            steps_this_epoch = 0

            utilized_codebook = set()

            for imgs in pbar:
                imgs = imgs.to(self.device)
                cur_batch_size = imgs.size(0)
                steps_this_epoch += cur_batch_size

                if self.use_amp:
                    with autocast(device_type=self.device):
                        decoded_images, _, vq_regularization_loss = self.model(imgs)

                        recon_loss = self.reconstruction_loss_fn(decoded_images, imgs)
                        loss = recon_loss + vq_regularization_loss
                    
                    self.opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    decoded_images, codebook_indices, vq_regularization_loss = self.model(imgs)
                    utilized_codebook.update(codebook_indices.tolist())

                    # perceptual_loss = torch.mean(self.perceptual_loss_fn(decoded_images, imgs))
                    recon_loss = self.reconstruction_loss_fn(decoded_images, imgs)
                    # loss = 0.5 * recon_loss + 0.5 * perceptual_loss + vq_regularization_loss
                    loss = recon_loss + vq_regularization_loss
                    
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                # should save some images out around here
                losses['loss'] += loss.item() * cur_batch_size
                losses['recon_loss'] += recon_loss.item() * cur_batch_size
                losses['reg_loss'] += vq_regularization_loss.item() * cur_batch_size
                
                pbar.set_postfix({
                    'Loss' : f"{losses['loss'] / steps_this_epoch:.5f}",
                    'Recon_Loss' : f"{losses['recon_loss'] / steps_this_epoch:.5f}",
                    'VQ_Reg_Loss' : f"{losses['reg_loss'] / steps_this_epoch:.5f}"
                })

            losses = {k : (v / self.len_data) for k, v in losses.items()}
            print(f"Codebook utilization: {len(utilized_codebook) / self.config['codebook_size']}")
            
            if self.logger.update_losses(losses, epoch):
                torch.save({
                    'epoch' : epoch,
                    'best_loss' : self.logger.get_best_loss(),
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.opt.state_dict()
                    }, os.path.join(self.out_dir, f"best_{epoch}.pt"))
            torch.save({
                    'epoch' : epoch,
                    'best_loss' : self.logger.get_best_loss(),
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.opt.state_dict()
                    }, os.path.join(self.out_dir, f"latest_{epoch}.pt"))
            
        self.logger.write_logs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, help = "name of config file in configs/", required = True)
    parser.add_argument("--load_checkpoint", action = 'store_true', required = False)
    parser.add_argument("--save_as", type = str, default = None, required = False)
    parser.add_argument("--verbose", action = 'store_true', required = False)
    args = parser.parse_args()

    config, out_dir, writer, checkpoint = utils.prepare_result_folder(args)

    try:
        trainer = VQVAETrainer(config, out_dir, writer)
        if args.load_checkpoint:
            trainer.load_checkpoint(checkpoint)
        trainer.train()
        writer.close()
    except KeyboardInterrupt:
        writer.close()

if __name__ == "__main__":
    main()
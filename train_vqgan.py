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
from models.discriminator import Discriminator
from losses import adopt_generator_weight, calculate_adaptive_weight, calculate_d_loss

class VQGANTrainer:
    def __init__(self, config, out_dir, summary_writer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = VQVAE(config)
        self.generator.to(self.device)

        self.discriminator = Discriminator(config)
        self.discriminator.to(self.device)

        # self.reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.perceptual_loss_fn = lpips.LPIPS(net = 'vgg').to(device = self.device)

        self.discriminator_weight = config['discriminator']['discriminator_weight']
        self.perceptual_weight = config['vqvae']['perceptual_weight']
        self.codebook_weight = config['vqvae']['codebook_weight']

        self.config = config
        self.num_epochs = config['num_epochs']
        self.start_epoch = 0
        self.out_dir = out_dir

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr = self.config['lr']
        )

        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr = self.config['lr'] # Using same LR for both rn
        )

        self.use_amp = config['use_amp']
        if self.use_amp:
            # self.scaler = GradScaler()
            raise NotImplementedError

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
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])

        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

    def save_checkpoint(self, epoch, name):
        torch.save({
            'epoch' : epoch,
            'best_loss' : self.logger.get_best_loss(),
            'generator_state_dict' : self.generator.state_dict(),
            'opt_g_state_dict' : self.opt_g.state_dict(),
            'discriminator_state_dict' : self.discriminator.state_dict(),
            'opt_d_state_dict' : self.opt_d.state_dict()
            }, os.path.join(self.out_dir, name))
    
    def train(self):
        train_dataloader = self.load_data()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} : {self.num_epochs}")
            losses = {'loss': 0,
                      'recon_loss': 0,
                      'commit_loss': 0}
            steps_this_epoch = 0

            utilized_codebook = set()

            for imgs in pbar:
                imgs = imgs.to(self.device)
                cur_batch_size = imgs.size(0)
                steps_this_epoch += cur_batch_size

                reconstructed_images, codebook_indices, commitment_loss = self.generator(imgs)
                utilized_codebook.update(codebook_indices.tolist())

                recon_loss = self.reconstruction_loss_fn(reconstructed_images, imgs)
                perceptual_loss = torch.mean(self.perceptual_loss_fn(reconstructed_images, imgs))
                # Not really sure why this is just a x + ay and not a (1-a)x + ax for 0 <= a <= 1. But this is what VQGAN does
                perc_rec_loss = recon_loss + self.perceptual_weight(perceptual_loss) 

                logits_fake = self.discriminator(reconstructed_images)
                logits_real = self.discriminator(imgs)
                g_loss = -torch.mean(logits_fake) # This is a shortcut implementation of non-saturating loss
                adaptive_weight = calculate_adaptive_weight(perc_rec_loss, 
                                                     g_loss, 
                                                     self.generator.get_last_layer_weights(), 
                                                     discriminator_weight=self.discriminator_weight)
                d_loss = calculate_d_loss(logits_fake = logits_fake,
                                          logits_real = logits_real)
                cur_step = epoch * self.config['batch_size'] + steps_this_epoch
                disc_factor = adopt_generator_weight(d_loss,
                                                     cur_step,
                                                     threshold = self.config['discriminator']['adopt_d_loss_step'],
                                                     value = 0)
                
                loss = perc_rec_loss + disc_factor * adaptive_weight * g_loss + self.codebook_weight * commitment_loss
                
                self.opt_g.zero_grad()
                loss.backward()
                self.opt_g.step()

                self.opt_d.zero_grad()
                d_loss.backward()
                self.opt_d.step()

                # should save some images out around here
                losses['loss'] += loss.item() * cur_batch_size
                losses['recon_loss'] += recon_loss.item() * cur_batch_size
                losses['commit_loss'] += commitment_loss.item() * cur_batch_size
                
                pbar.set_postfix({
                    'Loss' : f"{losses['loss'] / steps_this_epoch:.5f}",
                    'Recon_Loss' : f"{losses['recon_loss'] / steps_this_epoch:.5f}",
                    'Commit_Loss' : f"{losses['commit_loss'] / steps_this_epoch:.5f}"
                })

            losses = {k : (v / self.len_data) for k, v in losses.items()}
            print(f"Codebook utilization: {len(utilized_codebook) / self.config['vqae']['codebook_size']}")
            
            if self.logger.update_losses(losses, epoch):
                self.save_checkpoint(epoch, f"best_{epoch}.pt")
            self.save_checkpoint(epoch, f"latest_{epoch}.pt")
            
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
        trainer = VQGANTrainer(config, out_dir, writer)
        if args.load_checkpoint:
            trainer.load_checkpoint(checkpoint)
        trainer.train()
        writer.close()
    except KeyboardInterrupt:
        writer.close()

if __name__ == "__main__":
    main()
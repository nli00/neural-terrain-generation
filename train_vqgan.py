import torch
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2

import os
from tqdm import tqdm
import argparse
import lpips

import utils
from dataset import STL10Dataset, USGSDataset
from models.vqvae import VQVAE
from models.discriminator import Discriminator
from losses import adopt_generator_weight, calculate_adaptive_weight, bce_loss, hinge_loss

import torch.nn.functional as F

class VQGANTrainer:
    def __init__(self, config, out_dir, logger):
        self.dataset = config['dataset']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = VQVAE(config)
        self.generator.to(self.device)

        self.discriminator = Discriminator(config)
        self.discriminator.to(self.device)

        # self.reconstruction_loss_fn = torch.nn.L1Loss(reduction='mean')
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.perceptual_loss_fn = lpips.LPIPS(net = 'vgg').to(device = self.device)

        self.perceptual_weight = config['vqvae']['perceptual_weight']
        self.codebook_weight = config['vqvae']['codebook_weight']
        self.discriminator_weight = config['discriminator']['discriminator_weight']
        self.disc_factor = config['discriminator']['disc_factor']
        self.adopt_d_loss_step = config['discriminator']['adopt_d_loss_step']

        self.config = config
        self.num_epochs = config['num_epochs']
        self.start_epoch = 0
        self.out_dir = out_dir

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr = self.config['vqvae']['lr']
        )

        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr = self.config['discriminator']['lr']
        )

        self.use_amp = config['use_amp']
        if self.use_amp:
            # self.scaler = GradScaler()
            raise NotImplementedError

        os.makedirs("results", exist_ok = True)
        os.makedirs("checkpoints", exist_ok = True)

        self.logger = logger

    def load_data(self):
        training_data_means = self.config['means']
        training_data_stds = self.config['stds']

        # TODO: replace PIL pipeline fully with torch tensors for performance
        # So that means all this casing is temporary
        if self.dataset == 'STL10':
            training_transforms = v2.Compose(
                [v2.Resize((self.config['resolution'], self.config['resolution'])),
                v2.ToTensor(),
                v2.Normalize(mean=training_data_means, std=training_data_stds)
            ])
        elif self.dataset == 'USGS':
            training_transforms = v2.Compose([
                v2.Resize((self.config['resolution'], self.config['resolution'])),
                v2.Normalize(mean=training_data_means, std=training_data_stds)
            ])
        else:
            raise NotImplementedError

        if self.dataset == 'STL10':
            train_dataset = STL10Dataset(root_dir = self.config['train_dataset'], transform = training_transforms)
        elif self.dataset == 'USGS':
            train_dataset = USGSDataset(root_dir = self.config['train_dataset'], transform = training_transforms)
        else:
            raise NotImplementedError
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
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.opt_g.load_state_dict(checkpoint['opt_g_state_dict'])

        # If I want to implement LR adjustment for fine-tuning later or something
        # for param_group in self.opt_g.param_groups:
        #     param_group['lr'] = .00005

        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.opt_d.load_state_dict(checkpoint['opt_d_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']

        self.logger.load_logs(self.start_epoch)

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
            self.generator.train()
            self.discriminator.train()

            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} : {self.num_epochs}")
            losses = {'loss': 0,
                      'recon_loss': 0,
                      'commit_loss': 0,
                      'perceptual_loss': 0,
                      'disc_loss': 0,
                      'generator_loss': 0,
                      'disc_factor': 0,
                      'adaptive_weight': 0}
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
                # Not really sure why this is just x + ay and not (1-a)x + ay for 0 <= a <= 1. But this is what VQGAN does
                perc_rec_loss = recon_loss + self.perceptual_weight * perceptual_loss 

                logits_fake_gen = self.discriminator(reconstructed_images)
                # g_loss = -torch.mean(logits_fake_gen) # This is a shortcut implementation of non-saturating loss
                g_loss = -torch.mean(torch.log(torch.sigmoid(logits_fake_gen))) # Full implementation of non-saturating loss
                adaptive_weight = calculate_adaptive_weight(perc_rec_loss, 
                                                     g_loss, 
                                                     self.generator.get_last_layer_weights(), 
                                                     discriminator_weight=self.discriminator_weight)
                cur_step = epoch * self.len_data + steps_this_epoch
                disc_factor = adopt_generator_weight(self.disc_factor,
                                                     cur_step,
                                                     threshold = self.adopt_d_loss_step,
                                                     value = 0)

                logits_real = self.discriminator(imgs.contiguous().detach())
                logits_fake = self.discriminator(reconstructed_images.contiguous().detach())
                d_loss = hinge_loss(logits_fake = logits_fake,
                                          logits_real = logits_real)
                
                # TODO: Move this functionality over the the logger. This is useful to determine on a more granulary basis what is happening when the discriminator is doing well or poorly
                with open('logits.csv', mode = 'a') as f:
                    f.write(f'{torch.mean(logits_real.clone().detach()).item()},{torch.mean(logits_fake.clone().detach()).item()}\n')
                
                d_power_factor = max(0, 1 - d_loss)
                loss = perc_rec_loss + disc_factor * adaptive_weight * g_loss * d_power_factor + self.codebook_weight * commitment_loss

                self.opt_g.zero_grad()
                loss.backward(retain_graph=True)

                self.opt_d.zero_grad()
                d_loss.backward()

                self.opt_g.step()
                self.opt_d.step()

                losses['loss'] += loss.detach().item() * cur_batch_size
                losses['recon_loss'] += recon_loss.detach().item() * cur_batch_size
                losses['commit_loss'] += commitment_loss.detach().item() * cur_batch_size
                losses['perceptual_loss'] += perceptual_loss.detach().item() * cur_batch_size
                losses['disc_loss'] += d_loss.detach().item() * cur_batch_size
                losses['generator_loss'] += g_loss.detach().item() * cur_batch_size
                losses['disc_factor'] += disc_factor * cur_batch_size
                losses['adaptive_weight'] += adaptive_weight.detach().item() * cur_batch_size
                
                pbar.set_postfix({
                    'Loss' : f"{losses['loss'] / steps_this_epoch:.5f}",
                    'Recon_Loss' : f"{losses['recon_loss'] / steps_this_epoch:.5f}",
                    'Commit_Loss' : f"{losses['commit_loss'] / steps_this_epoch:.5f}"
                })

            losses = {k : (v / self.len_data) for k, v in losses.items()}
            losses['codebook_utilization'] = len(utilized_codebook) / self.config['vqvae']['codebook_size']
            
            if self.logger.update_losses(losses, epoch):
                self.save_checkpoint(epoch, f"best_{epoch}.pt")
            self.save_checkpoint(epoch, f"latest_{epoch}.pt")

            if ((epoch + 1) % 50 == 0):
                self.save_checkpoint(epoch, f"checkpoint_{epoch}.pt")

            self.logger.write_logs()
            
        self.logger.write_logs()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, help = "If training from 0: name of config file in configs. If resuming training, name of checkpoint dir", required = True)
    parser.add_argument("--load_checkpoint", action = 'store_true', required = False)
    parser.add_argument("--checkpoint", type = str, required = False)
    parser.add_argument("--save_as", type = str, default = None, required = False)
    parser.add_argument("--overwrite_ok", action = 'store_true', required = False)
    args = parser.parse_args()

    config, out_dir, writer, checkpoint = utils.prepare_result_folder(args)
    logger = utils.Logger(writer, out_dir)

    # TODO: TODO: move this over the the logger or get rid of it. This initializes some files for the adaptive weight function to log grads into,
    # but it really should be done by the logger so that it doesn't get overwritten each time the training runs.
    with open("gan_grad.csv", mode = 'w') as f:
        f.write('grad,g_loss\n')

    with open("rec_grad.csv", mode = 'w') as f:
        f.write('grad,rec_loss\n')

    with open("logits.csv", mode = 'w') as f:
        f.write('r,f\n')

    try:
        trainer = VQGANTrainer(config, out_dir, logger)
        if args.load_checkpoint:
            trainer.load_checkpoint(checkpoint)
        trainer.train()
        writer.close()
    except KeyboardInterrupt:
        writer.close()

if __name__ == "__main__":
    main()
from numpy.ma import masked_singleton
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image, make_grid

import os

import numpy as np
from tqdm import tqdm

from models.transformer import Transformer
from models.vqvae import Decoder
from dataset import EncodedImageDataset
import argparse
import yaml

from masking import outpaint_right
from scheduler import cosine_schedule

class Masked_Generator():
    def __init__(self, config, checkpoint_path):
        self.config = config

        self.transformer_checkpoint = checkpoint_path

        # Load the config for VQGAN components
        with open(self.config['vqgan_config'], 'r') as f:
            self.vqgan_config = yaml.safe_load(f)

        self.mask_token_id = self.vqgan_config['vqvae']['codebook_size']

        if torch.cuda.is_available:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            raise RuntimeWarning("Failed to detect CUDA capable GPU; defaulting to CPU")

        self.model = Transformer(config)
        self.model.to(self.device)

    def load_data(self, data: EncodedImageDataset):
        # dataloader = torch.utils.data.DataLoader(
        #     data,
        #     batch_size = self.config["batch_size"],
        #     shuffle = False,
        #     num_workers = self.config["num_workers"],
        #     pin_memory = True
        # )

        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size = 64,
            shuffle = False,
            num_workers = self.config["num_workers"],
            pin_memory = True
        )

        return dataloader

    def visualize(self, reconstructions, nrow=None):
        """
        Un-normalize images and produce visualization assuming that reconstructions has a depth of 1
        """
        
        std = self.vqgan_config['stds'][0]
        mean = self.vqgan_config['means'][0]

        reconstructions = reconstructions * std + mean # un normalize the images

        if nrow is None:
            nrow = reconstructions.shape[0]

        grid = make_grid(
            tensor=reconstructions, 
            nrow=nrow, 
            padding=0, 
            pad_value=1.0
            )

        return grid

    def mask_git_iterative_sampling(self, input_ids, mask, n_steps = 8):
        prediction_region_length = torch.sum(mask, dim = -1, keepdim = True)
        
        cumulative_patches = input_ids.clone()

        for i in range(n_steps):
            with torch.no_grad():
                logits = self.model(cumulative_patches) # B x L x C

            probs = F.softmax(logits, dim = -1) # B x L x C
            idx = torch.argmax(logits, dim = -1) # B x L

            # Let idx contain the predicted value for each of the masked patches, and the known value for each of the unmasked patches
            unknown_patches = (cumulative_patches == self.mask_token_id)
            idx = torch.where(unknown_patches, idx, cumulative_patches) 

            # * Technically, we could add a condition to exit early here on the last iter because everything should be predicted already, and we're not masking anything else
            
            confidences = torch.gather(probs, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1) # Unsqueeze because probs is BLC and idx is BL. Squeeze to remove added dim
            confidences = torch.where(unknown_patches, confidences, float('inf')) # Set the confidences of our known regions to infinity so we don't mask them

            # Section 3.2 step 3 https://arxiv.org/pdf/2202.04200
            n_mask = torch.ceil(cosine_schedule(torch.tensor((i + 1) / n_steps)) * prediction_region_length).int().to(device = logits.device)
            unknown_patch_count = torch.sum(unknown_patches[0])
            n_mask = torch.clamp(n_mask, n_mask.new_tensor(1), unknown_patch_count - 1) # minus 1 because we want to keep at least 1. by the way, this forces one random token to get masked on the last iter, but this is fine because we return idx anyway, not cumulative_patches

            # TODO: Add gumbel noise and temperature annealing to improve generation diversity

            sorted_confidences, _ = torch.sort(confidences, dim = -1)
            cut_off = torch.gather(sorted_confidences, dim = -1, index = n_mask) # get the value of the item in the sorted array at the n_mask index for each sequence
            iterative_mask = confidences < cut_off

            idx = torch.where(unknown_patches, idx, cumulative_patches) 
            cumulative_patches = torch.where(iterative_mask, self.mask_token_id, idx)

        return idx

    def outpaint(self, data: EncodedImageDataset, mask_proportion = .5):
        """
        evaluate model outpainting. Currently only equipped to outpaint one step to the right
        """
        dataloader = self.load_data(data)
        self.model.load_checkpoint(self.transformer_checkpoint)
        
        self.model.eval()

        idx = next(iter(dataloader))

        idx = idx.to(self.device, non_blocking=True).long() # Cast to long here because idx were saved as smaller data

        input_ids, mask = outpaint_right(
            idx,
            self.mask_token_id,
            mask_proportion
        )

        # logits = self.model(input_ids) # B x L x C
        # predicted_indices = torch.argmax(logits, dim=-1) # B x L

        # final_tokens = torch.where(mask, input_ids, predicted_indices) # Force the unmasked regions back to their original values

        final_tokens = self.mask_git_iterative_sampling(input_ids, mask, n_steps = 8)

        return final_tokens, idx # B x L, B x L

    def autoregressive_outpaint(self, idx, mask_proportion = .5):
        input_ids, mask = outpaint_right(
            idx,
            self.mask_token_id,
            mask_proportion
        )

        # logits = self.model(input_ids) # B x L x C
        # predicted_indices = torch.argmax(logits, dim=-1) # B x L

        # final_tokens = torch.where(mask, input_ids, predicted_indices) # Force the unmasked regions back to their original values

        final_tokens = self.mask_git_iterative_sampling(input_ids, mask, n_steps = 8)

        return final_tokens # B x L

    def decode(self, predicted_indices, patch_dimensions=None):
        """
        Decode predicted indices to an image. 
        predicted_indices: B x L
        patch_dimensions: optional (in units of patches). If None, assumes square (H=W). Otherwise, uses specified height and width.
        """
        checkpoint_path = self.config['vqgan_checkpoint']
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        # Load the codebook weights
        codebook_weight = ckpt['generator_state_dict']['vq_layer.codebook.weight']
        codebook_size, codebook_dim = codebook_weight.shape
        codebook = nn.Embedding(codebook_size, codebook_dim).to(self.device)
        codebook.weight.data.copy_(codebook_weight.to(self.device))

        # Load the decoder
        decoder_state = {k[len("decoder."):]: v for k, v in ckpt['generator_state_dict'].items() if k.startswith("decoder.")}
        decoder = Decoder(self.vqgan_config)
        decoder.load_state_dict(decoder_state)
        decoder = decoder.to(self.device)
        decoder.eval()

        # Look up corresponding embeddings in the codebook
        # predicted_indices: B x L
        B, L = predicted_indices.shape

        # Flatten indices in case they're not contiguous
        flat_indices = predicted_indices.view(-1)  # (B*L,)
        embeddings = codebook(flat_indices)        # (B*L, C)

        # Unflatten to B x L x C
        embeddings = embeddings.view(B, L, codebook_dim) # (B, L, C)

        # Determine proper height/width
        if patch_dimensions is not None:
            H, W = patch_dimensions
            assert H * W == L, f"Provided height*width ({H}*{W}={H*W}) does not match sequence length L={L}"
        else:
            # Default to square
            H = W = int(L ** 0.5)
            assert H * W == L, "Sequence length L must be a perfect square to reshape to 2D grid if no height/width provided"

        embeddings = embeddings.permute(0, 2, 1) # B x C x L
        embeddings = embeddings.view(B, codebook_dim, H, W) # B x C x H x W

        # Pass through decoder
        with torch.no_grad():
            reconstructions = decoder(embeddings)

        return reconstructions

    def generate(self, data):
        generated_indices, original_indices = self.outpaint(data)
        generated_reconstructions = self.decode(generated_indices)
        original_reconstructions = self.decode(original_indices)

        # Two-row visualization: generated samples on top, originals on bottom.
        stacked = torch.cat([generated_reconstructions, original_reconstructions], dim=0)
        images = self.visualize(stacked, nrow=generated_reconstructions.shape[0])

        return images

    def generate_2(self, data):
        divisor = 4
        mask_proportion = 1/divisor
        indices, _ = self.outpaint(data)
        indices_2 = self.autoregressive_outpaint(indices, mask_proportion=mask_proportion)
        indices_3 = self.autoregressive_outpaint(indices_2, mask_proportion=mask_proportion)
        indices_4 = self.autoregressive_outpaint(indices_3, mask_proportion=mask_proportion)
        indices_5 = self.autoregressive_outpaint(indices_4, mask_proportion=mask_proportion)
        indices_6 = self.autoregressive_outpaint(indices_5, mask_proportion=mask_proportion)
        indices_7 = self.autoregressive_outpaint(indices_6, mask_proportion=mask_proportion)

        # Concatenate indices, indices_2, and indices_3 horizontally, with 50% overlap between each pair

        # All inputs: [B, L] where L = H * W (and assumed square), so can reshape to [B, H, W] with H = W = sqrt(L)
        B, L = indices.shape
        H = W = int(L ** 0.5)
        assert H * W == L, "Sequence length L must be a perfect square"

        x1 = indices.view(B, H, W)
        x2 = indices_2.view(B, H, W)
        x3 = indices_3.view(B, H, W)
        x4 = indices_4.view(B, H, W)
        x5 = indices_5.view(B, H, W)
        x6 = indices_6.view(B, H, W)
        x7 = indices_7.view(B, H, W)

        assert W % divisor == 0, "Width must be divisible by divisor"

        n_masked_patches = W // divisor

        concat = torch.cat([
            x1,                   
            x2[:, :, W - n_masked_patches:],                   
            x3[:, :, W - n_masked_patches:],
            x4[:, :, W - n_masked_patches:],                   
            x5[:, :, W - n_masked_patches:],
            x6[:, :, W - n_masked_patches:],                   
            x7[:, :, W - n_masked_patches:],                    
        ], dim=2)  # concatenate along width

        patch_dimensions = concat.shape[1:]

        concat = concat.view(concat.shape[0], -1)

        reconstructions = self.decode(concat, patch_dimensions=patch_dimensions)
        images = self.visualize(reconstructions)

        return images

"""

"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, help = "path to directory with checkpoint", required = True)
    parser.add_argument("--checkpoint", type = str, required = True)
    parser.add_argument("--test_dataset", type = str, help = "path to .pt file of preprocessed images (generated by transformer_preprocessing.py)")
    args = parser.parse_args()

    torch.manual_seed(1423)

    with open(os.path.join("checkpoints", args.checkpoint_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    data = EncodedImageDataset(args.test_dataset)
    
    checkpoint_path = os.path.join("checkpoints", args.checkpoint_dir, args.checkpoint)

    generator = Masked_Generator(config, checkpoint_path)

    images = generator.generate(data)
    # images = generator.generate_2(data)

    # Saves image out in ./results with name corresponding the checkpoint used
    out_path = os.path.join("results", f"{args.checkpoint_dir}_{args.checkpoint[:-3]}")
    os.mkdir(out_path)

    save_image(
        images, 
        fp = os.path.join(out_path, f"output.png")
    )

# python3 masked_generation.py --checkpoint_dir transformer --checkpoint best_99.pt --test_dataset data/codebook_idx/usgs_d_power_test_geo_idx.pt
if __name__ == "__main__":
    main()
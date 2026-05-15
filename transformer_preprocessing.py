"""
Encode a dataset with the VQ-VAE encoder + vector quantizer only; save per-image codebook indices.
"""
import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
import yaml
from torch.utils.data import Dataset
from torchvision.transforms import v2

from dataset import STL10Dataset, USGSDataset
from models.vqvae import Encoder, VectorQuantizer


def load_data(means, stds, dataset_class: str, dataset_root: str, resolution, batch_size: int = 64):
    data_class = dataset_class.lower()
    if data_class == "stl10":
        training_transforms = v2.Compose(
            [
                v2.Resize((resolution, resolution)),
                v2.ToTensor(),
                v2.Normalize(mean=means, std=stds),
            ]
        )
        data = STL10Dataset(root_dir=dataset_root, transform=training_transforms)
    elif data_class == "usgs":
        training_transforms = v2.Compose(
            [
                v2.Resize((resolution, resolution)),
                v2.Normalize(mean=means, std=stds),
            ]
        )
        data = USGSDataset(root_dir=dataset_root, transform=training_transforms)
    else:
        raise ValueError(f"dataset_kind must be stl10 or usgs; got {data_class!r}")

    # Shuffle is False because we want to be able to link the codebook indices back to the original images later for qualitative assessment
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return dataloader


def indices_storage_dtype(codebook_size: int) -> torch.dtype:
    """Smallest unsigned integer type that can hold indices in [0, codebook_size)."""
    if codebook_size <= 256:
        return torch.uint8
    if codebook_size <= 65536:
        return torch.uint16
    if codebook_size <= 2**32:
        return torch.uint32
    return torch.int64


class EncoderVQ(nn.Module):
    """Encoder + VQ layer only (matches prefix stripping from full VQVAE checkpoints)."""

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.vq_layer = VectorQuantizer(
            codebook_size=config["vqvae"]["codebook_size"],
            codebook_dim=config["vqvae"]["latent_dim"],
        )

    def load_encoder_vq_from_checkpoint(self, path: str, map_location=None):
        raw = torch.load(path, map_location=map_location)
        gen_state = raw["generator_state_dict"]
        encoder_state = {k[len("encoder."):]: v for k, v in gen_state.items() if k.startswith("encoder.")}
        vq_state = {k[len("vq_layer."):]: v for k, v in gen_state.items() if k.startswith("vq_layer.")}
        self.encoder.load_state_dict(encoder_state)
        self.vq_layer.load_state_dict(vq_state)

@torch.no_grad()
def collect_codebook_indices(model: EncoderVQ, dataloader, device: torch.device, storage_dtype: torch.dtype):
    dataset_len = len(dataloader.dataset)
    num_patches = None
    out = None
    offset = 0

    model.eval()
    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        z_e = model.encoder(batch)
        _, flat_indices, _ = model.vq_layer(z_e)
        b, _, h, w = z_e.shape
        patches_per_image = h * w
        idx_2d = flat_indices.view(b, patches_per_image)

        if num_patches is None:
            num_patches = patches_per_image
            out = torch.empty((dataset_len, num_patches), dtype=storage_dtype, device="cpu")

        out[offset : offset + b].copy_(idx_2d.cpu().to(storage_dtype))
        offset += b

    assert offset == dataset_len
    return out, num_patches

"""
Example usage:
python3 transformer_preprocessing.py --checkpoint_dir usgs_d_power --checkpoint best_98.pt --out_dir data/codebook_idx --dataset_root data/train_geo --dataset_class usgs
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory under checkpoints/ with config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename (e.g. model.pt)")
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output .pt directory",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Directory containing images (layout depends on --dataset_class)",
    )
    parser.add_argument(
        "--dataset_class",
        type=str,
        required=True,
        choices=("stl10", "usgs"),
        help="stl10: train_image_png_{i}.png | usgs: {i}.png",
    )
    args = parser.parse_args()

    checkpoint_dir = os.path.join("checkpoints", args.checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    dataloader = load_data(
        means=config["means"],
        stds=config["stds"],
        dataset_class=args.dataset_class,
        dataset_root=os.path.expanduser(args.dataset_root),
        resolution=config["resolution"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebook_size = int(config["vqvae"]["codebook_size"])
    storage_dtype = indices_storage_dtype(codebook_size)

    model = EncoderVQ(config).to(device)
    ckpt_path = os.path.join(checkpoint_dir, args.checkpoint)
    model.load_encoder_vq_from_checkpoint(ckpt_path, map_location=device)

    indices_2d, num_patches = collect_codebook_indices(model, dataloader, device, storage_dtype)

    out_file = f"{args.checkpoint_dir}_{os.path.basename(args.dataset_root)}_idx.pt"
    out_path = os.path.join(args.out_dir, out_file)
    payload = {
        # indices[i] is shape (num_patches,) — codebook index per latent position (row-major h*w)
        "indices": indices_2d,
        "num_patches": num_patches,
        "codebook_size": codebook_size,
        "storage_dtype": str(storage_dtype),
        "num_images": indices_2d.shape[0],
        "dataset_root": os.path.abspath(os.path.expanduser(args.dataset_root)),
        "dataset_class": args.dataset_class.lower(),
    }
    torch.save(payload, out_path)
    print(f"Saved {indices_2d.shape[0]} x {num_patches} indices ({storage_dtype}) to {out_path}")


if __name__ == "__main__":
    main()

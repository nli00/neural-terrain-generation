"""
Training loop for the bidirectional (MaskGIT-style) transformer on quantized latents.
"""
import yaml
import argparse
import glob
import os
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from dataset import EncodedImageDataset
from models.transformer import Transformer

from scheduler import cosine_schedule
from masking import mask_inputs

class TransformerTrainer:
    def __init__(
        self,
        config: dict,
        mask_scheduler,
        out_dir: Optional[str] = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cpu":
            raise RuntimeWarning("Model failed to load on GPU. Defaulting to CPU.")

        self.model = Transformer(config)
        self.model.to(self.device)

        lr = config["lr"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.num_epochs = config["num_epochs"]
        self.start_epoch = 0

        self.mask_scheduler = mask_scheduler

        self.out_dir = out_dir

        if os.path.exists(out_dir):
            print(f"Contining will overwrite existing checkpoints in {out_dir}.")
            response = input("Type \"continue\" to continue.\n")
            if response != "continue":
                return
        else:
            os.makedirs(self.out_dir)
            
        with open(os.path.join(out_dir, "config.yaml"), 'w') as f:
            yaml.dump(config, f)

        self.best_loss = float("inf")

    def load_data(self) -> DataLoader:
        indices_pt = self.config["indices_pt"]
        if not indices_pt:
            raise KeyError(
                "config['indices_pt'] must point to the .pt file produced by transformer_preprocessing.py (EncodedImageDataset)."
            )

        train_dataset = EncodedImageDataset(indices_pt)

        self.codebook_size = train_dataset.codebook_size
        self.mask_token_id = self.codebook_size

        seq_model = self.model.sequence_length
        if train_dataset.num_patches != seq_model:
            raise ValueError(
                f"Encoded sequence length {train_dataset.num_patches} does not match "
                f"Transformer.sequence_length {seq_model}. Regenerate indices or adjust the transformer parameters."
            )

        return DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

    def _remove_checkpoints(self, stem_prefix: str) -> None:
        if not self.out_dir:
            return
        pattern = os.path.join(self.out_dir, f"{stem_prefix}_*.pt")
        for path in glob.glob(pattern):
            os.remove(path)

    def save_checkpoint(self, epoch: int, epoch_loss: float) -> None:
        """
        Persist at most one `latest_<epoch>.pt` and one `best_<epoch>.pt` under ``out_dir``.
        When the current epoch is the best so far (including ties), only ``best_<epoch>.pt``
        is kept; otherwise ``latest_<epoch>.pt`` is written and the previous best file
        is left unchanged.
        """

        payload = {
            "epoch": epoch,
            "loss": epoch_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        is_best = epoch_loss <= self.best_loss
        if is_best:
            self.best_loss = epoch_loss
            self._remove_checkpoints("latest")
            self._remove_checkpoints("best")
            torch.save(
                payload,
                os.path.join(self.out_dir, f"best_{epoch}.pt"),
            )
        else:
            self._remove_checkpoints("latest")
            torch.save(
                payload,
                os.path.join(self.out_dir, f"latest_{epoch}.pt"),
            )

    def train(self) -> None:
        train_dataloader = self.load_data()
        self.model.train()

        for epoch in range(self.start_epoch, self.num_epochs):
            batch_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            total_epoch_loss = 0.0
            num_steps = 0

            for batch in batch_iter:
                batch = batch.to(self.device, non_blocking=True).long() # Cast to long here because inx were saved as smaller data
                cur_batch_size = batch.size(0)

                input_ids, labels = mask_inputs(
                    batch,
                    self.mask_token_id,
                    self.mask_scheduler,
                    ignore_index=-100,
                )

                assert not (labels == -100).all(), f"Batch is entirely unmasked. Check masking logic. {labels[labels != -100].shape, labels[labels == -100].shape}"

                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), # B * L, classes view
                    labels.view(-1), # B * L, index is class
                )

                total_epoch_loss += loss.item() * cur_batch_size
                num_steps += cur_batch_size
                avg_loss = total_epoch_loss / num_steps

                # Write average loss so far to progress bar
                # By the way, this is an average of averages since the CSE loss already reports a weighted average, so this reported loss
                # should be used for reference during training only, not as a metric
                batch_iter.set_postfix({'avg_loss': f"{avg_loss:.4f}"})

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            assert num_steps > 0
            avg_epoch_loss = total_epoch_loss / num_steps
            self.save_checkpoint(epoch, avg_epoch_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config name under configs/ (with or without .yaml), same style as train_vqgan.py.",
    )
    args = parser.parse_args()

    config, config_name = utils.read_config(args.config)

    # TODO: Replace this with logic for training, resuming training, and saving checkpoints consistant with train_vqgan.py
    out_dir = os.path.join("checkpoints", config_name)

    trainer = TransformerTrainer(config, mask_scheduler = cosine_schedule, out_dir=out_dir)
    trainer.train()


if __name__ == "__main__":
    main()

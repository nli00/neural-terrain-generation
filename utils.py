import yaml
import os
import glob
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import pandas as pd

# For some reason I decided that I needed to adjust for whether the command line arg
# for the config was specified as config or config.yaml. So this is a function to do that.
# Really this is just unecessary though.
def read_config(config : str):
    if not config.endswith(".yaml"):
        config_file = config + ".yaml"
        config_name = config
    else:
        config_file = config
        config_name = config[:-5]

    with open(os.path.join("configs", config_file)) as f:
        config = yaml.safe_load(f)

    return config, config_name

def prepare_result_folder(args : argparse.Namespace):
    config, config_name = read_config(args.config)
    if args.save_as:
        config_name = args.save_as
        
    out_dir = os.path.join("checkpoints", config_name)
    if args.load_checkpoint:
        try:
            checkpoint_path = glob.glob(os.path.join(out_dir, '*latest*.pt'))[0]
            checkpoint = torch.load(checkpoint_path)
            print(f"Continuing from saved checkpoint {checkpoint_path}\n")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return
    else:
        checkpoint = None

    if os.path.exists(os.path.join("checkpoints", config_name)):
        if not args.overwrite_ok:
            print(f"Contining will overwrite existing checkpoints in checkpoints/{config_name}.")
            response = input("Type \"continue\" to continue.\n")
            if response != "continue":
                return
    else:
        os.mkdir(os.path.join("checkpoints", config_name))
    
    with open(os.path.join("checkpoints", config_name, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(os.path.join("checkpoints", config_name, 'logs'))

    return config, out_dir, writer, checkpoint

class Logger():
    def __init__(self, writer, out_dir):
        self.best_losses = {}
        self.losses = {}
        self.writer = writer
        self.out_dir = out_dir
    
    def update_losses(self, losses : dict, epoch : int) -> bool:
        is_best = False

        for loss_type, value in losses.items():
            if loss_type not in self.losses.keys():
                self.losses[loss_type] = [value]
                self.best_losses[loss_type] = value
                if loss_type == 'loss':
                    is_best = True
            else:
                self.losses[loss_type].append(value)

                if value < self.best_losses[loss_type]:
                    self.best_losses[loss_type] = value
                    if loss_type == 'loss':
                        is_best = True
                        self.cleanup_checkpoints('best')

            self.writer.add_scalar(loss_type, value, epoch)

        self.cleanup_checkpoints('latest')
        self.writer.flush()
        return is_best
    
    def cleanup_checkpoints(self, type : str):
        checkpoints = glob.glob(os.path.join(self.out_dir, f"*{type}*.pt"))
        for checkpoint in checkpoints:
            os.remove(checkpoint)

    def get_best_loss(self) -> float:
        return self.losses['loss']
    
    def write_logs(self):
        losses = pd.DataFrame(self.losses)
        losses.to_csv(os.path.join(self.out_dir, 'losses.csv'))
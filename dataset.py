import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TerrainDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, idx)
        image = None # Load the image into a torch tensor

        return image, idx
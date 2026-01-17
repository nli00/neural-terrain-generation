import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os
import numpy as np

# class TerrainDataset(Dataset):
#     def __init__(self, img_dir):
#         self.img_dir = img_dir

#     def __len__(self):
#         return len(os.listdir(self.img_dir))
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, idx)
#         image = None # Load the image into a torch tensor

#         return image, idx

class STL10Dataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f"train_image_png_{idx + 1}.png"
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
    
class USGSDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f"{idx}.png"
        img_path = os.path.join(self.root_dir, img_name)
        image = torchvision.io.read_image(img_path) / 255
        img_float = torchvision.transforms.v2.functional.to_dtype(image, torch.float32, scale = False)

        if self.transform:
            image = self.transform(img_float)

        return image
    
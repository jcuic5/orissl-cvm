from os.path import join
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
import numpy as np


class ImagesFromList(Dataset):
    def __init__(self, img_paths, img_size, transform):
        self.img_paths = img_paths
        self.img_size = img_size
        self.transform = transform(self.img_size)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = self.transform(Image.open(self.img_paths[idx]).convert("RGB"))
        except (FileNotFoundError, OSError):
            img = self.transform(Image.new('RGB', (self.img_size[0], self.img_size[1])))
            
        return img, idx
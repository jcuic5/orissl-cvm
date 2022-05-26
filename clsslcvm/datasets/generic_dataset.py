from os.path import join
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
import numpy as np


class ImagesFromList(Dataset):
    def __init__(self, root_dir, images, transform, category='ground'):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try: img = self.transform(Image.open(join(self.path, self.images[idx])))
        except: return None
        return img, idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None

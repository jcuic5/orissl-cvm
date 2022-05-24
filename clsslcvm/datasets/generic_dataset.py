from os.path import join
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
import numpy as np


class ImagesFromList(Dataset):
    def __init__(self, root_dir, images, transform, category='ground'):
        self.root_dir = root_dir
        if category == 'ground':
            self.path = join(self.root_dir, 'streetview')
            self.key = 'img_gr'
        elif category == 'satellite':
            self.path = join(self.root_dir, 'polarmap')
            self.key = 'img_sa'
        else:
            raise NotImplementedError

        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = self.transform(Image.open(join(self.path, self.images[idx][self.key])))
        except:
            return None
        return img, idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None


class ImagePairsFromList(Dataset):
    def __init__(self, root_dir, images, transform):
        self.root_dir = root_dir
        self.gr_path = join(self.root_dir, 'streetview')
        self.sa_path = join(self.root_dir, 'polarmap')

        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_gr = self.transform(Image.open(join(self.gr_path, self.images[idx]['img_gr'])))
            img_sa = self.transform(Image.open(join(self.sa_path, self.images[idx]['img_sa'])))
        except:
            return None
        return img_gr, img_sa, idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None
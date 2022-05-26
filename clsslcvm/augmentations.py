import numpy as np
from scipy import rand
import torch
import random
from torchvision.transforms import GaussianBlur
from torchvision import transforms as T


imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

def input_transform(resize):
    return T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean_std[0], 
                    std=imagenet_mean_std[1]),
    ])


class InputPairTransform():
    def __init__(self, image_size):
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean_std[0], std=imagenet_mean_std[1]),
        ])
    def __call__(self, img_gr, img_sa):
        img_gr = self.transform(img_gr)
        img_sa = self.transform(img_sa)
        return img_gr, img_sa


class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size            
        p_blur = 0.5
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2,1.0)),
            # T.RandomCrop((image_size[0], int(image_size[1]/1.2))), # NOTE my setting
            # T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=[x//20*2+1 for x in image_size], sigma=(0.1,2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
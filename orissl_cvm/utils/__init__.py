import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.linalg as LA

imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

def input_transform(resize=(112, 616)):
    if resize[0] > 0 and resize[1] > 0:
        return T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean_std[0], 
                        std=imagenet_mean_std[1]),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean_std[0], 
                        std=imagenet_mean_std[1]),
        ])


def soft_triplet_loss(a, p, n, gamma=10.0):
    # print(f'ditance of a with p: {LA.norm(a - p)}, a with n: {LA.norm(a - n)}')
    loss = torch.log(1 + torch.exp(gamma * (LA.norm(a - p) - LA.norm(a - n))))
    return loss


def _initialize_weights(nn_model):
    for m in nn_model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 


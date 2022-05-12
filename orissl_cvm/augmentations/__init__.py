import numpy as np
from scipy import rand
import torch
import random
from torchvision.transforms import GaussianBlur
from torchvision import transforms as T


imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

def input_transform(resize=(112,616)):
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


def random_slide_pano(img, mode='cls', num_dirs=4):
    H, W = img.shape[-2], img.shape[-1]
    if mode == 'cls':
        label = random.randint(0, num_dirs - 1)
        slide_w = int(float(label / num_dirs) * W)
        # NOTE draw a dividing line for debug
        # img[..., :5] = 0
        img = np.concatenate([img[..., slide_w:], img[..., :slide_w]], axis=-1)
        # img[..., slide_w:slide_w+10] = 0
    elif mode == 'reg':
        # slide_w = random.randint(0, W - 1)
        # slide_w = random.randint(0, 15) * 32 # num of angles * num of pixels per step
        # NOTE for debug
        # slide_w = random.randint(0, 7) * 32
        slide_w = random.randint(0, 31) * 8
        # slide_w = random.randint(0, 1)
        # slide_w = slide_w * (W // 28 - 1) * 14
        # slide_w = 3 * 16

        # slide_w = random.random()
        # slide_w = int(slide_w * W // 2)
        # p = random.random()
        # if p > 0.5:
        #     img = np.concatenate([img[..., slide_w:], img[..., :slide_w]], axis=-1)
        #     label = (slide_w + W // 2) / W
        # else:
        #     img = np.concatenate([img[..., -slide_w:], img[..., :-slide_w]], axis=-1)
        #     label = - (slide_w + W // 2) / W

        img = np.concatenate([img[..., -slide_w:], img[..., :-slide_w]], axis=-1)
        label = slide_w / W + 1 # NOTE plus 1 for keep a positive value
        
    else:
        raise NotImplementedError

    return img, label


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


# version 1: relative orientation between ground and satellite
class OriLabelTransform():
    def __init__(self, image_size, num_dirs=4):
        self.num_dirs = num_dirs
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean_std[0], std=imagenet_mean_std[1]),
            lambda x : random_slide_pano(x, 'cls', num_dirs)
        ])
    def __call__(self, img_gr, img_sa):
        img_gr, l_gr = self.transform(img_gr)
        img_sa, l_sa = self.transform(img_sa)
        label = l_gr - l_sa if l_gr - l_sa >= 0 else l_gr - l_sa + self.num_dirs
        return img_gr, img_sa, label


# version 2: separate relative orientation between itselfs
class OriLabelPairTransform():
    def __init__(self, image_size, num_dirs=4):
        self.num_dirs = num_dirs
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean_std[0], std=imagenet_mean_std[1]),
        ])
        self.shift = lambda x : random_slide_pano(x, 'reg')
    def __call__(self, img_gr, img_sa):
        img_gr1, l_gr1 = self.shift(self.transform(img_gr))
        img_gr2, l_gr2 = self.shift(self.transform(img_gr))
        l1 = l_gr1 - l_gr2
        img_sa1, l_sa1 = self.shift(self.transform(img_sa))
        img_sa2, l_sa2 = self.shift(self.transform(img_sa))
        l2 = l_sa1 - l_sa2
        return img_gr1, img_gr2, l1, img_sa1, img_sa2, l2


class SimSiamTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std):
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size            
        p_blur = 0.5
        self.transform = T.Compose([
            # T.RandomResizedCrop(image_size, scale=(0.2,1.0)),
            T.RandomCrop((image_size[0], int(image_size[1]/1.2))), # NOTE my setting
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=[x//20*2+1 for x in image_size], sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
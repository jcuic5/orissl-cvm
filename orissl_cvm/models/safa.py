from turtle import forward
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SPE(nn.Module):
    def __init__(self, fmp_size=(7,38)):
        super(SPE, self).__init__()
        H, W = fmp_size
        self.fc1 = nn.Linear(H*W, H*W//2, bias=True)
        self.fc2 = nn.Linear(H*W//2, H*W, bias=True)

    def forward(self, fmp):
        B, C, H, W = fmp.shape
        # max pool
        fmp_pooled, _ = fmp.max(axis=-3, keepdim=False) #(B, H, W)
        # spatial-aware improtance generator
        x = fmp_pooled.flatten(start_dim=-2, end_dim=-1) #(B, H*W)
        x = self.fc2(self.fc1(x)).reshape(B, -1, H*W) #(B, D, H*W)
        # aggregate
        fmp = fmp.flatten(start_dim=-2, end_dim=-1) #(B, C, H*W)

        # feat = torch.einsum('bci,bdi->bdc', fmp_pooled, x) #(B, C, D)
        # feat = feat.flatten(start_dim=-2, end_dim=-1) #(B, C*D)
        # feat = F.normalize(feat, p=2, dim=1)

        feat = torch.mul(fmp, x).sum(dim=-1)

        return feat


class SAFAvgg16(nn.Module):
    def __init__(self):
        super(SAFAvgg16, self).__init__()
        # settings
        self.encoder_dim = 512
        self.num_spes = 8
        self.desc_dim = self.encoder_dim * self.spe_dim * self.num_spes #default: 4096

        self.nn_model_gr = self.get_model()
        self.nn_model_sa = self.get_model()

    def get_model(self):
        nn_model = nn.Module()
        nn_model.add_module('encoder', self.get_backend())
        spe_gr = nn.Module()
        for i in range(self.num_spes): 
            spe_gr.add_module(f'spe_{i}', SPE())
        nn_model.add_module('spe', spe_gr)
        
        return nn_model

    def get_backend(self):
        enc = models.vgg16(pretrained=True)
        # drop the last two layers: ReLU and MaxPool2d
        layers = list(enc.features.children())[:-2]
        # NOTE optionally freeze part of the backbone
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        # for layer in layers[:-5]:
        #     for p in layer.parameters():
        #         p.requires_grad = False
        enc = nn.Sequential(*layers)

        return enc

    def forward(self, x1, x2):
        descriptor = []
        for nn_model, x in zip((self.nn_model_gr, self.nn_model_sa), (x1, x2)):
            enc = nn_model.encoder(x)
            desc = torch.cat([spe_i(enc) for spe_i in nn_model.spe.children()], dim=1)
            descriptor.append(F.normalize(desc, p=2, dim=1))

        return tuple(descriptor)


class SAFAvgg16Cls(nn.Module):
    def __init__(self):
        super().__init__()

        self.SAFAvgg16 = SAFAvgg16()
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4, bias=True)
        )

    def forward(self, x1, x2):
        desc = torch.cat(self.SAFAvgg16(x1, x2), dim=1)
        return self.classifier(desc)
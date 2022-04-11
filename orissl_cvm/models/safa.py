import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SPE(nn.Module):
    def __init__(self):
        super(SPE, self).__init__()

        # pool size=14, dimension=8
        self.pool = nn.AdaptiveMaxPool2d((14, 14))
        self.fc1 = nn.Linear(14*14, int(14*14*8/2), bias=True)
        self.fc2 = nn.Linear(int(14*14*8/2), 14*14*8, bias=True)

    def forward(self, fmp):
        # max pool
        fmp_pooled = self.pool(fmp)
        B, C, H, W = fmp_pooled.shape
        # spatial-aware improtance generator
        x = torch.mean(fmp_pooled, dim=1).flatten(start_dim=-2, end_dim=-1)
        x = self.fc2(self.fc1(x)).reshape(B, -1, H*W) #(B, D, H*W)
        # aggregate
        fmp_pooled = fmp_pooled.flatten(start_dim=-2, end_dim=-1) #(B, C, H*W)
        feat = torch.einsum('bci,bdi->bdc', fmp_pooled, x) #(B, C, D): frobenius on each C, D
        feat = feat.flatten(start_dim=-2, end_dim=-1) #(B, C*D)

        return F.normalize(feat, p=2, dim=1)


class SAFAvgg16(nn.Module):
    def __init__(self):
        super(SAFAvgg16, self).__init__()
        # settings
        self.encoder_dim = 512
        self.spe_dim = 8
        self.num_spes = 1
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
        desc = []
        for nn_model, x in zip((self.nn_model_gr, self.nn_model_sa), (x1, x2)):
            enc = nn_model.encoder(x)
            desc.append(torch.cat([spe_i(enc) for spe_i in nn_model.spe.children()], dim=1))

        return tuple(desc)

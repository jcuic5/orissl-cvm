import imp
from turtle import forward, shape
from urllib import response
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from orissl_cvm.tools.visualize import visualize_desc
from orissl_cvm.models import get_backbone, get_pool, _initialize_weights


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


class SPEPool(nn.Module):
    def __init__(self, num_spe=8, norm=True):
        super(SPEPool, self).__init__()
        self.num_spe = num_spe
        for i in range(num_spe): 
            self.add_module(f'spe_{i}', SPE())
        self.norm = norm

    def forward(self, x):
        feat = torch.cat([spe_i(x) for spe_i in self.children()], dim=1)
        return F.normalize(feat, p=2, dim=1) if self.norm else feat


class CrossViewMatchingModel(nn.Module):
    def __init__(self, backbone, pool):
        super(CrossViewMatchingModel, self).__init__()
        # settings
        self.nn_model_gr = self.get_model(backbone, pool)
        self.nn_model_sa = self.get_model(backbone, pool)

    def get_model(self, backbone, pool):
        nn_model = nn.Module()
        nn_model.add_module('backbone', get_backbone(backbone))
        nn_model.add_module('pool', get_pool(pool))
        return nn_model

    def forward(self, x1, x2):
        descriptor = []
        for nn_model, x in zip((self.nn_model_gr, self.nn_model_sa), (x1, x2)):
            fmp = nn_model.backbone(x)
            feat = nn_model.pool(fmp)
            descriptor.append(feat)

        return tuple(descriptor)


class SAFAvgg16Cls(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_gr = self.get_backend()
        self.encoder_sa = self.get_backend()
        self.classifier = nn.Sequential(
            nn.Linear(38, 4, bias=True),
            # nn.Linear(4256, 4, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(4096, 1024, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(1024, 4, bias=True)
        )

    def get_backend(self):
        enc = models.vgg16(pretrained=True)
        # drop the last two layers: ReLU and MaxPool2d
        layers = list(enc.features.children())[:-2]
        # NOTE optionally freeze part of the backbone
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        layers[-5] = nn.Conv2d(512, 256, kernel_size=3, stride=(1, 1), padding=1)
        layers[-3] = nn.Conv2d(256, 64, kernel_size=3, stride=(1, 1), padding=1)
        layers[-1] = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        for layer in layers[-5:]:
            _initialize_weights(layer)
        enc = nn.Sequential(*layers)
        return enc

    def ori_corr(self, fmp_gr, fmp_sa):
        assert fmp_gr.shape[:3] == fmp_sa.shape[:3]
        W_gr, W_sa = fmp_gr.shape[-1], fmp_sa.shape[-1]
        fmp_sa = torch.cat([fmp_sa[..., -(W_gr//2-1):], fmp_sa, fmp_sa[..., :W_gr//2]], dim=-1)
        resp = F.conv2d(fmp_sa, fmp_gr, bias=None, stride=1, padding=0)
        assert resp.shape[-2] == 1 and resp.shape[-1] == W_sa
        resp = resp.mean(dim=-3, keepdim=False).flatten(start_dim=-2, end_dim=-1)
        return resp

    def forward(self, x1, x2):

        fmp_gr, fmp_sa = self.encoder_gr(x1), self.encoder_sa(x2)

        # NOTE orientation response like DSM
        resp = self.ori_corr(fmp_gr, fmp_sa)
        output = self.classifier(resp)

        # NOTE cated fmp
        # visualize_desc(fmp_gr.flatten(start_dim=1)[:8], fmp_sa.flatten(start_dim=1)[:8])
        # fmp_fused = torch.cat([fmp_gr, fmp_sa], dim=-3)
        # fmp_fused = torch.flatten(fmp_fused, start_dim=1)
        # output = self.classifier(fmp_fused)

        # NOTE stacked input
        # x = torch.cat([x1, x2], dim=-2)
        # fmp_fused = self.encoder_gr(x)
        # visualize_desc(fmp_fused.flatten(start_dim=1)[:8], fmp_fused.flatten(start_dim=1)[:8])
        # fmp_fused = torch.flatten(fmp_fused, start_dim=1)
        # output = self.classifier(fmp_fused)
        
        return output
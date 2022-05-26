from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F

from clsslcvm.tools import visualize
from .__init__ import get_backbone, get_pool, _initialize_weights
from .simsiam import SimSiam
from .vit import ViT
from clsslcvm.tools import show_cam_on_image


def horizontal_correlation(fmp1, fmp2):
    '''Find feature map's maximum correlation horizontal position on another

    args: 
        fmp1: query feature map (will slide on fmp2) - (B, C, H, W1)
        fmp2: reference feature map - (B, C, H, W2)
    return:
        resp: the horizontal index on fmp2 where fmp1 has the maximum response
    '''
    assert fmp1.shape[:3] == fmp2.shape[:3]
    B, C, H = fmp1.shape[:3]
    W1, W2 = fmp1.shape[3], fmp2.shape[3]
    fmp2 = torch.cat([fmp2, fmp2[..., :(W1 - 1)]], dim=-1)
    # NOTE the batch dim of the kernel fmp will serve as the num of kernels. so now do it separately
    resp = torch.cat([F.conv2d(fmp2[i:i+1], fmp1[i:i+1], bias=None, stride=1, padding=0) for i in range(B)], dim=0) # (B, 1, 1, W2)
    resp = resp.flatten(start_dim=-3, end_dim=-1)
    _, resp = torch.max(resp, dim=1, keepdim=True)
    return resp


class SPE(nn.Module):
    def __init__(self, fmp_size): # for vgg16, 616 -> 38, 512 -> 32, for resnet18, 512 -> 16
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
    def __init__(self, fmp_size, num_spe=8):
        super(SPEPool, self).__init__()
        self.num_spe = num_spe
        for i in range(num_spe): 
            self.add_module(f'spe_{i}', SPE(fmp_size))

    def forward(self, x):
        feat = torch.cat([spe_i(x) for spe_i in self.children()], dim=1)
        return feat


class CVMModel(nn.Module):
    def __init__(self, backbone, pool):
        super(CVMModel, self).__init__()
        self.features_gr = get_backbone(backbone)
        self.features_sa = get_backbone(backbone)
        self.pool_gr = get_pool(pool, norm=True)
        self.pool_sa = get_pool(pool, norm=True)

    def forward(self, x1, x2):
        fmp_gr = self.features_gr(x1)
        desc_gr = self.pool_gr(fmp_gr)
        fmp_sa = self.features_sa(x2)
        desc_sa = self.pool_sa(fmp_sa)
        # fmap_gr = fmp_gr.cpu().data.numpy().squeeze()
        # fmap_sa = fmp_sa.cpu().data.numpy().squeeze()
        # visualize.visualize_assets(x1, torch.tensor(fmap_gr).mean(1), x2, torch.tensor(fmap_sa).mean(1))
        # visualize.visualize_assets(show_cam_on_image(x1, torch.tensor(fmap_gr).mean(1)), show_cam_on_image(x2, torch.tensor(fmap_sa).mean(1)))
        # visualize.visualize_assets(desc_gr, desc_sa, mode='descriptor')

        return desc_gr, desc_sa
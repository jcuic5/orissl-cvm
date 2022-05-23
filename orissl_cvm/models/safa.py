from cv2 import norm
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F

from orissl_cvm.tools import visualize
from .__init__ import get_backbone, get_pool, _initialize_weights
from .simsiam import SimSiam
from .vit import ViT
from orissl_cvm.tools import show_cam_on_image


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


class CrossViewMatchingModel(nn.Module):
    def __init__(self, backbone, pool, shared=False):
        super(CrossViewMatchingModel, self).__init__()
        self.shared = shared
        if not shared:
            self.features_gr = get_backbone(backbone)
            self.features_sa = get_backbone(backbone)
        else:
            self.features = get_backbone(backbone)
        self.pool = get_pool(pool, norm=True)

    def forward(self, x1, x2):
        if not self.shared:
            fmp_gr = self.features_gr(x1)
            desc_gr = self.pool(fmp_gr)
            fmp_sa = self.features_sa(x2)
            desc_sa = self.pool(fmp_sa)
        else:
            fmp_gr = self.features(x1)
            desc_gr = self.pool(fmp_gr)
            fmp_sa = self.features(x2)
            desc_sa = self.pool(fmp_sa)
        # fmap_gr = fmp_gr.cpu().data.numpy().squeeze()
        # fmap_sa = fmp_sa.cpu().data.numpy().squeeze()
        # visualize.visualize_assets(x1, torch.tensor(fmap_gr).mean(1), x2, torch.tensor(fmap_sa).mean(1))
        # visualize.visualize_assets(show_cam_on_image(x1, torch.tensor(fmap_gr).mean(1)), show_cam_on_image(x2, torch.tensor(fmap_sa).mean(1)))
        # visualize.visualize_assets(desc_gr, desc_sa, mode='descriptor')

        return desc_gr, desc_sa


class CrossViewMatchingModelv2(nn.Module):
    def __init__(self, backbone, pool, shared=False):
        super(CrossViewMatchingModelv2, self).__init__()
        self.shared = shared
        if not shared:
            self.features_gr = get_backbone(backbone)
            self.features_sa = get_backbone(backbone)
            self.pool_gr = get_pool(pool, norm=True)
            self.pool_sa = get_pool(pool, norm=True)
        else:
            self.features = get_backbone(backbone)
            self.pool = get_pool(pool, norm=True)

    def forward(self, x1, x2):
        if not self.shared:
            fmp_gr = self.features_gr(x1)
            desc_gr = self.pool_gr(fmp_gr)
            fmp_sa = self.features_sa(x2)
            desc_sa = self.pool_sa(fmp_sa)
        else:
            fmp_gr = self.features(x1)
            desc_gr = self.pool(fmp_gr)
            fmp_sa = self.features(x2)
            desc_sa = self.pool(fmp_sa)
        # fmap_gr = fmp_gr.cpu().data.numpy().squeeze()
        # fmap_sa = fmp_sa.cpu().data.numpy().squeeze()
        # visualize.visualize_assets(x1, torch.tensor(fmap_gr).mean(1), x2, torch.tensor(fmap_sa).mean(1))
        # visualize.visualize_assets(show_cam_on_image(x1, torch.tensor(fmap_gr).mean(1)), show_cam_on_image(x2, torch.tensor(fmap_sa).mean(1)))
        # visualize.visualize_assets(desc_gr, desc_sa, mode='descriptor')

        return desc_gr, desc_sa


class CrossViewOriPredModel(nn.Module):
    def __init__(self, backbone, pool, shared=False):
        super(CrossViewOriPredModel, self).__init__()
        self.shared = shared
        if not shared:
            self.features_gr = get_backbone(backbone)
            self.features_sa = get_backbone(backbone)
        else:
            self.features = get_backbone(backbone)
        self.pool = get_pool(pool, norm=False)

        # self.classifier = nn.Linear(1, 1, bias=True)
        # nn.init.constant_(self.classifier.weight, 1)
        # nn.init.constant_(self.classifier.bias, 0)
        self.classifier = nn.Sequential(
                           nn.Linear(1024, 256),
                           nn.ReLU(),
                           nn.Dropout(),
                           nn.Linear(256, 1))

    def forward(self, x1, x2, x3, x4):
        if not self.shared:
            fmp1 = self.features_gr(x1)
            fmp2 = self.features_gr(x2)
            d1 = self.pool(fmp1)
            d2 = self.pool(fmp2)
            d12 = torch.cat([d1, d2], dim=-1)
            fmp3 = self.features_sa(x3)
            fmp4 = self.features_sa(x4)
            d3 = self.pool(fmp3)
            d4 = self.pool(fmp4)
            d34 = torch.cat([d3, d4], dim=-1)
        else:
            d1 = self.pool(self.features(x1))
            d2 = self.pool(self.features(x2))
            d12 = torch.cat([d1, d2], dim=-1)
            d3 = self.pool(self.features(x3))
            d4 = self.pool(self.features(x4))
            d34 = torch.cat([d3, d4], dim=-1)

        visualize.visualize_assets(fmp1.mean(axis=1, keepdim=True), fmp2.mean(axis=1, keepdim=True), mode='image')
        visualize.visualize_assets(fmp3.mean(axis=1, keepdim=True), fmp4.mean(axis=1, keepdim=True), mode='image')
        visualize.visualize_assets(d1, d2, d3, d4, mode='descriptor')
        output12 = self.classifier(d12)
        output34 = self.classifier(d34)
        # NOTE 
        # 1. label制定规则为x1相对于x2向右移了多少角度比例（向左也全部等效到了向右）
        # 2. horizontal_correlation计算fmp1在fmp2上的哪个水平位置上响应最大
        # 所以预测和该函数输出直接存在一个反方向，例如如果fmp1在fmp2的第二个位置上响应最大，说明
        # 其本来相对于fmp2是向左移动了1个位置，因此这里做了一点处理
        # visualize.visualize_assets(d1.mean(axis=1, keepdim=True), d2.mean(axis=1, keepdim=True), mode='image')
        # output12 = (1 - horizontal_correlation(d1, d2) / d2.shape[-1]).float()
        # output34 = (1 - horizontal_correlation(d3, d4) / d4.shape[-1]).float()
        # output12 = self.classifier(output12)
        # output34 = self.classifier(output34)
        return output12, output34
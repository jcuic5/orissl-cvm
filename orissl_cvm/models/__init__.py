from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, vgg16


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


def _initialize_weights(m):
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


def get_backbone(name):
    backbone = eval(f"{name}()")
    if name == 'resnet18' or name == 'resnet50':
        layers = list(backbone.children())[:-2]
        backbone = nn.Sequential(*layers)
    elif name == 'vgg16':
        # drop the last two layers: ReLU and MaxPool2d
        layers = list(backbone.features.children())[:-2]
        # NOTE optionally freeze part of the backbone
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        # for layer in layers[:-5]:
        #     for p in layer.parameters():
        #         p.requires_grad = False
        backbone = nn.Sequential(*layers)
    else:
        raise NotImplementedError

    return backbone


def get_pool(name, norm=True):
    from .safa import SPEPool

    if name == 'max':
        return nn.Sequential(*[nn.AdaptiveMaxPool2d((1, 1)), Flatten(), L2Norm()]) if norm else \
               nn.Sequential(*[nn.AdaptiveMaxPool2d((1, 1)), Flatten()])
    elif name == 'avg':
        return nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), Flatten(), L2Norm()]) if norm else \
               nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), Flatten()])
    elif name == 'safa':
        return SPEPool(8, norm)
    else:
        raise NotImplementedError


def get_model(model_cfg):
    from .simsiam import SimSiam
    from .byol import BYOL
    from .simclr import SimCLR
    from .safa import CrossViewMatchingModel

    feat_dim = 4096 if model_cfg.pool == 'safa' else 512

    if model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone), get_pool(model_cfg.pool, norm=False), feat_dim)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    elif model_cfg.name == 'crossview':
        model = CrossViewMatchingModel(model_cfg.backbone, model_cfg.pool)
    else:
        raise NotImplementedError

    return model

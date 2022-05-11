import imp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16, mobilenet_v2
from .vit import ViT

class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


class LayerNorm(nn.Module):
    def forward(self, input_data):
        return F.layer_norm(input_data, normalized_shape=input_data.shape[-3:])


class MinusIdentity(nn.Module):
    def forward(self, input_data):
        return - input_data


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
    if name == 'resnet18':
        # backbone = resnet18(norm_layer=lambda x : nn.BatchNorm2d(x, track_running_stats=False))
        # backbone = resnet18(norm_layer=lambda x : nn.Identity())
        backbone = resnet18(norm_layer=lambda x : LayerNorm())
        layers = list(backbone.children())[:-2]
        # NOTE optionally if we have dim less than 224, don't half the size at conv_1
        # layers[0].stride = 1
        layers[0] = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        backbone = nn.Sequential(*layers)
    elif name == 'vgg16' or name == 'mobilenet_v2':
        backbone = eval(f"{name}()")
        # drop the last two layers: ReLU and MaxPool2d
        layers = list(backbone.features.children())[:-1]
        # layers = [x for x in layers if not isinstance(x, nn.ReLU)]

        # optionally freeze part of the backbone
        # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
        # for layer in layers[:-5]:
        #     for p in layer.parameters():
        #         p.requires_grad = False
        # layers[-5] = nn.Conv2d(512, 256, kernel_size=3, stride=(1, 1), padding=1)
        # layers[-3] = nn.Conv2d(256, 64, kernel_size=3, stride=(1, 1), padding=1)
        # layers[-1] = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        # for layer in layers[-5:]:
        #     _initialize_weights(layer)
        backbone = nn.Sequential(*layers)
    elif name == 'vit':
        backbone = ViT(
            image_size=(112, 512),
            patch_size=(16, 16),
            num_classes=512,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif name == 'identity':
        backbone = nn.Identity()
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
    elif name == 'min':
        return nn.Sequential(*[MinusIdentity(), nn.AdaptiveMaxPool2d((1, 1)), Flatten(), L2Norm()]) if norm else \
               nn.Sequential(*[MinusIdentity(), nn.AdaptiveMaxPool2d((1, 1)), Flatten()])
    elif name == 'safa':
        return SPEPool(fmp_size=(7, 32), num_spe=1, norm=norm)
    elif name == 'identity':
        return nn.Identity()
    elif name == 'fc':
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 8)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(28672, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 512)
        )
    else:
        raise NotImplementedError


def get_encoder(backbone, pool):
    nn_model = nn.Module()
    nn_model.add_module('backbone', backbone)
    nn_model.add_module('pool', pool)
    return nn_model


def get_model(model_cfg):
    from .simsiam import SimSiam
    from .safa import CrossViewMatchingModel, CrossViewOriPredModel
    feat_dim = 4096 if model_cfg.pool == 'safa' else 512
    if model_cfg.name == 'simsiam':
        model = SimSiam(model_cfg.backbone, model_cfg.pool, model_cfg.proj_layers, feat_dim)
    elif model_cfg.name == 'cvm':
        model = CrossViewMatchingModel(model_cfg.backbone, model_cfg.pool, shared=model_cfg.shared)
    elif model_cfg.name == 'oripred':
        model = CrossViewOriPredModel(model_cfg.backbone, model_cfg.pool, shared=model_cfg.shared)
    else:
        raise NotImplementedError

    return model


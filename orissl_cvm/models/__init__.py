import imp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16
from .vit import ViT


class L2Norm(nn.Module):
    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=1)


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
        backbone.output_dim = 512
    elif name == 'vgg16':
        # backbone = eval(f"{name}()")
        backbone = vgg16(pretrained=True)
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
        backbone.output_dim = 512
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
        backbone.output_dim = 512
    elif name == 'identity':
        backbone = nn.Identity()
        backbone.output_dim = 3
    else:
        raise NotImplementedError

    return backbone


def get_pool(name, norm=True):
    from .safa import SPEPool
    if name == 'max':
        pool = [nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten(start_dim=1, end_dim=-1)]
    elif name == 'avg':
        pool = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1, end_dim=-1)]
    elif name == 'safa':
        pool = [SPEPool(fmp_size=(7, 32), num_spe=1)]
    elif name == 'identity':
        pool = [nn.Identity()]
    elif name == 'fc':
        pool = [nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(25088, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 512)]
    else:
        raise NotImplementedError
    if norm: 
        pool.append(L2Norm())
    return nn.Sequential(*pool)


def get_model(model_cfg):
    from .simsiam import SimSiam
    from .safa import CrossViewMatchingModel, CrossViewOriPredModel
    if model_cfg.name == 'simsiam':
        model = SimSiam(model_cfg.backbone)
    elif model_cfg.name == 'cvm':
        model = CrossViewMatchingModel(model_cfg.backbone, model_cfg.pool, shared=model_cfg.shared)
    elif model_cfg.name == 'oripred':
        model = CrossViewOriPredModel(model_cfg.backbone, model_cfg.pool, shared=model_cfg.shared)
    else:
        raise NotImplementedError

    return model


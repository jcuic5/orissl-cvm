'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


def get_backend():
    enc_dim = 512
    enc = models.vgg16(pretrained=True)
    layers = list(enc.features.children())[:-2]
    # NOTE: optional
    # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
    # for layer in layers[:-5]:
    #     for p in layer.parameters():
    #         p.requires_grad = False
    enc = nn.Sequential(*layers)
    return enc_dim, enc


def get_model(encoder, encoder_dim, config):
    # config['global_params'] is passed as config
    nn_model = nn.Module()
    nn_model.add_module('encoder', encoder)

    if config['pooling'].lower() == 'max':
        global_pool = nn.AdaptiveMaxPool2d((1, 1))
        nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif config['pooling'].pooling.lower() == 'avg':
        global_pool = nn.AdaptiveAvgPool2d((1, 1))
        nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    else:
        raise ValueError('Unknown pooling type: ' + config['pooling'].lower())

    return nn_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


def get_pool(config): 
    # config['global_params'] is passed as config
    if config['pooling'].lower() == 'max':
        global_pool = nn.AdaptiveMaxPool2d((1, 1))
        return nn.Sequential(*[global_pool, Flatten(), L2Norm()])
    elif config['pooling'].lower() == 'avg':
        global_pool = nn.AdaptiveAvgPool2d((1, 1))
        return nn.Sequential(*[global_pool, Flatten(), L2Norm()])
    else:
        raise ValueError('Unknown pooling type: ' + config['pooling'].lower())


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
        feat = torch.einsum('bci,bdi->bdc', fmp_pooled, x) #(B, C, D) - frobenius on each C,D
        feat = feat.flatten(start_dim=-2, end_dim=-1) #(B, C*D)

        return F.normalize(feat, p=2, dim=1)


class SAFAvgg16(nn.Module):
    def __init__(self, config):
        super(SAFAvgg16, self).__init__()

        self.nn_model_gr = self.get_model()
        self.nn_model_sa = self.get_model()

        self.encoder_dim = 512
        self.spe_dim = 8
        self.num_spes = 1
        self.desc_dim = self.encoder_dim * self.spe_dim * self.num_spes #default: 4096

    def get_model(self):
        nn_model = nn.Module()
        nn_model.add_module('encoder', self.get_backend())
        spe_gr = nn.Module()
        for i in range(1): 
            spe_gr.add_module(f'spe_{i}', SPE())
        nn_model.add_module('spe', spe_gr)
        
        return nn_model

    def get_backend(self):
        enc = models.vgg16(pretrained=True)
        # NOTE optionally drop the last two layers: ReLU and MaxPool2d
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

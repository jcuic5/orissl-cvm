import torch.nn as nn
import torch.nn.functional as F
from orissl_cvm.models.models_generic import get_backend, get_model


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


class GoodNet(nn.Module):
    def __init__(self, config):
        super(GoodNet, self).__init__()

        encoder_dim_gr, encoder_gr = get_backend()
        self.nn_model_gr = get_model(encoder_gr, encoder_dim_gr, config)
        encoder_dim_sa, encoder_sa = get_backend()
        self.nn_model_sa = get_model(encoder_sa, encoder_dim_sa, config)

        # NOTE: now desc dims from two sources are expected to be the same
        assert(encoder_dim_gr == encoder_dim_sa)
        self.encoder_dim = encoder_dim_gr

    def forward(self, x1, x2):
        desc_gr = self.nn_model_gr.pool(self.nn_model_gr.encoder(x1))
        desc_sa = self.nn_model_sa.pool(self.nn_model_sa.encoder(x2))
        return desc_gr, desc_sa

    def get_embedding(self, x1, x2):
        return self.nn_model_gr.encoder(x1), self.nn_model_sa.encoder(x2)

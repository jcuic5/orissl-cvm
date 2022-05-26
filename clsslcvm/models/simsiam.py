import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from .__init__ import get_backbone, get_pool


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class SimSiam(nn.Module):
    def __init__(self, backbone, pool):
        super().__init__()
        
        self.features = get_backbone(backbone)
        self.pool = get_pool(pool, norm=False)
        self.projector = projection_MLP(7*7*self.features.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.features,
            self.pool,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}


class SimSiamv2(nn.Module):
    def __init__(self, backbone, pool):
        super().__init__()
        '''Not use projector. Simply use the pool consistent with downstream CVM'''
        self.features = get_backbone(backbone)
        self.pool = get_pool(pool, norm=False)
        self.encoder = nn.Sequential( # f encoder
            self.features,
            self.pool
        )
        self.predictor = prediction_MLP(in_dim=512)
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}


class SimSiamv3(nn.Module):
    def __init__(self, backbone, pool):
        super().__init__()
        '''Joint training two views. Separate backbone, shared projector and predictor'''
        
        self.features_gr, self.features_sa = get_backbone(backbone), get_backbone(backbone)
        self.pool_gr, self.pool_sa = get_pool(pool, norm=False), get_pool(pool, norm=False)
        self.projector = projection_MLP(7*7*self.features_gr.output_dim)

        self.encoder_gr = nn.Sequential( # f encoder
            self.features_gr,
            self.pool_gr,
            self.projector
        )
        self.encoder_sa = nn.Sequential( # f encoder
            self.features_sa,
            self.pool_sa,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2, x3, x4):

        f1, f2, h = self.encoder_gr, self.encoder_sa, self.predictor
        z1, z2 = f1(x1), f1(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        z3, z4 = f2(x3), f2(x4)
        p3, p4 = h(z3), h(z4)
        L += D(p3, z4) / 2 + D(p4, z3) / 2
        return {'loss': L}
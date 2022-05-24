import torch
import torch.linalg as LA


class SoftTripletLoss(object):
    def __init__(self):
        self.gamma = 10.0

    def __call__(self, a, p, n):
        # print(f'ditance of a with p: {LA.norm(a - p)}, a with n: {LA.norm(a - n)}')
        loss = torch.log(1 + torch.exp(self.gamma * (LA.norm(a - p) - LA.norm(a - n))))
        return loss


def cycle_mse_loss(x, y):
    loss1 = (x - y).abs()
    loss2 = (x - (1 - y)).abs()
    loss = torch.minimum(loss1, loss2)
    return loss.sum()


def align_loss(x, y, alpha=2):
    '''From https://github.com/SsnL/align_uniform'''
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    '''From https://github.com/SsnL/align_uniform'''
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
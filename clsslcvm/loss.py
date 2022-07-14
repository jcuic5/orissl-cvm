import torch
import torch.linalg as LA


class SoftTripletLoss(object):
    def __init__(self):
        self.gamma = 10.0

    def __call__(self, a, p, n):
        # print(f'ditance of a with p: {LA.norm(a - p)}, a with n: {LA.norm(a - n)}')
        loss = torch.log(1 + torch.exp(self.gamma * (LA.norm(a - p) - LA.norm(a - n))))
        return loss


class SoftTripletLossv2(object):
    '''
    Compute the weighted soft-margin triplet loss (a PyTorch version)
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''
    def __init__(self):
        self.gamma = 10.0

    def __call__(self, sat_global, grd_global, batch_hard_count, batch_size):
        dist_array = 2 - 2 * (sat_global @ grd_global.T)
        pos_dist = torch.diagonal(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * self.gamma))) / pair_n
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * self.gamma))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = torch.log(1 + torch.exp(triplet_dist_g2s * self.gamma))
            top_k_g2s, _ = torch.top_k(torch.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = torch.sum(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            triplet_dist_s2g = torch.log(1 + torch.exp(triplet_dist_s2g * self.gamma))
            top_k_s2g, _ = torch.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = torch.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

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
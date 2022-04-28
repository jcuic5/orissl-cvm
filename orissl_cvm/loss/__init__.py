import torch
import torch.linalg as LA


class SoftTripletLoss(object):
    def __init__(self):
        self.gamma = 10.0

    def __call__(self, a, p, n):
        # print(f'ditance of a with p: {LA.norm(a - p)}, a with n: {LA.norm(a - n)}')
        loss = torch.log(1 + torch.exp(self.gamma * (LA.norm(a - p) - LA.norm(a - n))))
        return loss
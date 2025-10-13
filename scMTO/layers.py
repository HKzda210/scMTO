import torch
import torch.nn as nn
import torch.nn.functional as F

class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()
    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0, largescale=False):
        eps = 1e-10
        if largescale: scale_factor = 1.0
        mean = mean * scale_factor
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2
        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result

class TOPICLoss(nn.Module):
    def __init__(self):
        super(TOPICLoss, self).__init__()
    def forward(self, x, x_rec, V, L, lg=1e-6, ls=0):
        self.model = 'MSE'
        if self.model == 'MSE':
            result = F.mse_loss(x, x_rec)
        elif self.model == 'KLD':
            x = torch.clamp(x, min=1e-10)
            x_rec = torch.clamp(x_rec, min=1e-10)
            result = torch.mean(x * torch.log(x / x_rec) - x + x_rec)
        elif self.model == 'Cross_Entropy':
            result = torch.mean(-x * torch.log(torch.clamp(F.softmax(x_rec, dim=1), min=1e-2, max=1.0)))
        regularization = 0.0
        if lg > 0: # graph regularization if needed
            if isinstance(L, torch.sparse.Tensor) and L.layout == torch.sparse_coo:
                regularization += lg* torch.trace(V.T @ torch.sparse.mm(L, V))
            else:
                regularization += lg* torch.trace(V.T @ L @ V)
        if ls > 0: # sparse regularization if needed
            regularization += torch.sum(torch.norm(V, p=1, dim=1))
        result += regularization
        return result


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class GaussianNoise(nn.Module):
    def __init__(self, sigma=1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    def forward(self, x):
        return x + self.sigma * torch.randn_like(x)



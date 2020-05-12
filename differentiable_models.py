import torch
import torch.nn as nn
from band_models import One_band
from neural_nets import ThreeToOneNN


class Differentiable_one_band(nn.Module, One_band):
    def __init__(self, eta=0.5, **kw):
        super().__init__()
        One_band.__init__(self, **kw)
        self.t = nn.Parameter(torch.Tensor([self.t]))
        self.tp = nn.Parameter(torch.Tensor([self.tp]))
        self.tpp = nn.Parameter(torch.Tensor([self.tpp]))
        self.mu = nn.Parameter(torch.Tensor([self.mu]))
        self.eta = nn.Parameter(torch.Tensor([eta]))
    def forward(self, kx, ky, w):
        return self.spectral_weight(kx, ky, w, eta=torch.abs(self.eta))


class NeuralSelf(nn.Module):
    def __init__(self, eta=0.05):
        super().__init__()
        self.delta = ThreeToOneNN()
        self.xi = ThreeToOneNN()
        self.eta = eta

    def forward(self, kx, ky, w):
        delta_sq = self.delta(kx, ky, w)**2
        diff_w_xi = w - self.xi(kx, ky, w)
        eta_sq = self.eta**2
        re = delta_sq * diff_w_xi /(diff_w_xi**2 + eta_sq)
        im = delta_sq * eta_sq /(diff_w_xi**2 + eta_sq)
        return re, im


# class Differentiable_spectral_weight_with_self(nn.Module):
#     def __init__(self, tp0=0.0, tpp0=0.0, mu0=0.0, eta0=1.0):
#         super().__init__()
#         self.tp = nn.Parameter(torch.Tensor([tp0]))
#         self.tpp = nn.Parameter(torch.Tensor([tpp0]))
#         self.mu = nn.Parameter(torch.Tensor([mu0]))
#         self.eta = nn.Parameter(torch.Tensor([eta0]))
#         self.neural_self = NeuralSelf()

#     def forward(self, kx, ky, w):
#         disp = dispersion(kx, ky, mu=self.mu, t=1, tp=self.tp, tpp=self.tpp)
#         re_sigma, im_sigma = self.neural_self(kx, ky, w)
#         eta = torch.abs(self.eta)
#         out = spectral_weight(w, disp, eta, re_sigma, im_sigma)
#         return out

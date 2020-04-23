# Copyright (c) Simon Verret, 2020 
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def dispersion(kx, ky, mu=-1, t=1, tp=-0.3, tpp=0.2):
    coskx = torch.cos(kx)
    cosky = torch.cos(ky)
    cos2kx = torch.cos(2*kx)
    cos2ky = torch.cos(2*ky)
    
    out = -2*t*(coskx + cosky)
    out -= 4*tp*coskx*cosky 
    out -= 2*tpp*(cos2kx + cos2ky) 
    out -= mu 
    return out


def spectral_weight(omega, energy, eta=0.05):
    return eta / ((omega - energy)**2 + eta**2)


def print_fkw(function_of_kx_ky_w, w=0, save_path=None):
    kx = torch.linspace(0,np.pi,101)
    ky = torch.linspace(0,np.pi,101)
    kxx, kyy = torch.meshgrid(kx, ky)
    Akw_on_mesh = function_of_kx_ky_w(kxx,kyy,w)


    plt.figure(figsize=(4, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax.set_aspect('equal', 'box')
    ax.pcolormesh(
        np.transpose(kx),
        np.transpose(ky),
        np.transpose(Akw_on_mesh.detach()),
    )
    if save_path is not None: 
        plt.savefig(save_path)
    plt.show()

target_Akw = lambda kx,ky,w: spectral_weight(w, dispersion(kx,ky))
print_fkw(target_Akw)


#%% differentiable spectral weight

class DifferentiableSW(nn.Module):
    def __init__(self, t0=1, tp0=-0.4, tpp0=0.1, mu0=-1, eta0=1.0):
        super().__init__()
        self.t = nn.Parameter(torch.Tensor([t0]))
        self.tp = nn.Parameter(torch.Tensor([tp0]))
        self.tpp = nn.Parameter(torch.Tensor([tpp0]))
        self.mu = nn.Parameter(torch.Tensor([mu0]))
        self.eta = nn.Parameter(torch.Tensor([eta0]))

    def forward(self, kx, ky, w ):
        disp = dispersion(kx, ky, mu=self.mu, t=self.t, tp=self.tp, tpp=self.tpp)
        out = spectral_weight(w, disp, eta=self.eta)
        return out


diff_Akw = DifferentiableSW()
print_fkw(diff_Akw)



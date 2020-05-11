# Copyright (c) Simon Verret, 2020 
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.animation import FuncAnimation
import argparse

default_args = {
    'lr' : 0.05,
    'loss' : 'JS',
    'schedule' : '',
    'factor' : 0.9,
    'plot' : True,
    'plot_loss' : True,
    'animate' : False,
    'optim' : 'Adam',
    'batch_size' : 1000,
    'epochs': 500
}

parser = argparse.ArgumentParser()
for name, val in default_args.items():
    if type(val) is list:
        parser.add_argument("--"+name, nargs="+", type=type(val[0]), default=val)
    elif type(val) is bool and val is True:
        parser.add_argument("--no_"+name, dest=name, action="store_false")
    elif type(val) is bool and val is False:
        parser.add_argument("--"+name, action="store_true", default=val)
    elif val is None:
        parser.add_argument("--"+name, default=val)
    else:
        parser.add_argument("--"+name, type=type(val), default=val)
args = parser.parse_known_args()[0]


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


def spectral_weight(omega, energy, eta=0.05, re_sigma=0, im_sigma=0):
    return eta / ((omega - energy + re_sigma)**2 + (eta + im_sigma)**2)


def print_fkw(function_of_kx_ky_w, w=0, save_path=None, idx=0):
    kx = torch.linspace(0,np.pi,101).float()
    ky = torch.linspace(0,np.pi,101).float()
    kxx, kyy = torch.meshgrid(kx, ky)

    ## for compatibility with neural nets
    kxx = kxx.unsqueeze(-1)
    kyy = kyy.unsqueeze(-1)
    w = (w*torch.ones_like(kxx)).float()
    
    Akw_on_mesh = function_of_kx_ky_w(kxx,kyy,w)
    if isinstance(Akw_on_mesh, tuple):
        Akw_on_mesh = Akw_on_mesh[idx]

    ## for compatibility with neural nets
    Akw_on_mesh = Akw_on_mesh.squeeze()

    fig = plt.figure(figsize=(4, 6), dpi=80, facecolor='w', edgecolor='k')
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


def YRZ_sigma(kx, ky, omega, delta=0.2, eta=0.05):
    xi0 = dispersion(kx, ky, tp=0, tpp=0, mu=0)
    re_sigma = delta**2 * (omega - xi0) / ((omega - xi0)**2 + (eta)**2)
    im_sigma = delta**2 * eta / ((omega - xi0)**2 + (eta)**2)
    return re_sigma, im_sigma


def target_Akw(kx, ky, w, eta=0.1, delta=0.3):
    re_sigma, im_sigma = YRZ_sigma(kx,ky,w, delta, eta)
    xi = dispersion(kx,ky)
    return spectral_weight(w, xi, eta, re_sigma, im_sigma)


print('target')
print_fkw(target_Akw)
print_fkw(YRZ_sigma, idx=0)
print_fkw(YRZ_sigma, idx=1)


class Differentiable_spectral_weight(nn.Module):
    def __init__(self, tp0=0.0, tpp0=0.0, mu0=0.0, eta0=1.0):
        super().__init__()
        self.tp = nn.Parameter(torch.Tensor([tp0]))
        self.tpp = nn.Parameter(torch.Tensor([tpp0]))
        self.mu = nn.Parameter(torch.Tensor([mu0]))
        self.eta = nn.Parameter(torch.Tensor([eta0]))

    def forward(self, kx, ky, w ):
        disp = dispersion(kx, ky, mu=self.mu, t=1, tp=self.tp, tpp=self.tpp)
        out = spectral_weight(w, disp, eta=torch.abs(self.eta))
        return out


class ThreeToOneNN(nn.Module):
    def __init__(self, hsize=16):
        super().__init__()
        self.linear1 = nn.Linear(3,hsize)
        self.linear2 = nn.Linear(hsize,1)

        ## initialization of weights
        nn.init.xavier_uniform(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        nn.init.xavier_uniform(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)

    def forward(self, kx, ky, w):
        cat = torch.stack([kx,ky,w], dim=-1)
        out = F.relu(self.linear1(cat))
        out = F.relu(self.linear2(out))
        out = out.squeeze(-1)
        return out


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


class Differentiable_spectral_weight_with_self(nn.Module):
    def __init__(self, tp0=0.0, tpp0=0.0, mu0=0.0, eta0=1.0):
        super().__init__()
        self.tp = nn.Parameter(torch.Tensor([tp0]))
        self.tpp = nn.Parameter(torch.Tensor([tpp0]))
        self.mu = nn.Parameter(torch.Tensor([mu0]))
        self.eta = nn.Parameter(torch.Tensor([eta0]))
        self.neural_self = NeuralSelf()

    def forward(self, kx, ky, w):
        disp = dispersion(kx, ky, mu=self.mu, t=1, tp=self.tp, tpp=self.tpp)
        re_sigma, im_sigma = self.neural_self(kx, ky, w)
        eta = torch.abs(self.eta)
        out = spectral_weight(w, disp, eta, re_sigma, im_sigma)
        return out


model_Akw = Differentiable_spectral_weight_with_self()
# print('starting model')
# print_fkw(model_Akw)

def kl_div(Amodel, Atarget):
    """Kullback-Leibler Divergence"""
    return (Amodel*Amodel.log() - Amodel*Atarget.log()).mean()

def js_div(Amodel, Atarget):
    """Jensen-Shannone Divergence"""
    Amiddle = (Amodel + Atarget)/2
    return (kl_div(Amodel, Amiddle) + kl_div(Atarget, Amiddle))/2

def manual_loss(Amodel, Atarget):
    return (Atarget*Atarget.log() - Atarget*Amodel.log() - Amodel*Atarget.log()).mean()


if args.optim == 'adam' or args.optim == 'Adam':
    optimizer = torch.optim.Adam(model_Akw.parameters(), lr=args.lr, weight_decay=0.1)
elif args.optim == 'sgd' or args.optim == 'SGD':
    optimizer = torch.optim.SGD(model_Akw.parameters(), lr=args.lr)


print('training')
model_history = []
loss_history = []
for batch in range(1,args.epochs):
    # sample a thousand random (kx,ky,w) points
    kx_batch = torch.FloatTensor(args.batch_size).uniform_(0, np.pi)
    ky_batch = torch.FloatTensor(args.batch_size).uniform_(0, np.pi)
    w_batch = torch.FloatTensor(args.batch_size).uniform_(-2, 2)

    optimizer.zero_grad()
    results = model_Akw(kx_batch,ky_batch,w_batch)
    targets = target_Akw(kx_batch,ky_batch,w_batch)

    # I checked this:
    # assert manual_KL(results,targets) -  F.kl_div(results.log(),targets) < 1e-4
    if args.loss == 'manual':
        loss = manual_loss(results,targets)
    elif args.loss in ['mse','MSE','l2','L2']:
        loss = F.mse_loss(results, targets)
    elif args.loss in ['mae','MAE','l1','L1']:
        loss = F.l1_loss(results, targets)
    elif args.loss in ['JS', 'js', 'Jesen-Shannon']:
        loss = js_div(results, targets)
    else :
        loss = F.kl_div(results.log(), targets)
    loss.backward()

    if args.schedule in ['loss','KL']:
        rate = args.lr * loss.item()
        for g in optimizer.param_groups: g['lr'] = rate
    elif args.schedule == 'exp':
        rate = args.lr * args.factor
        for g in optimizer.param_groups: g['lr'] = rate
    else:
        rate = args.lr
    loss_history.append(loss.item())

    optimizer.step()
    if batch%(args.epochs//60) == 1:
        model_history.append(deepcopy(model_Akw))
        print(f'  loss = {loss:8.4f}    lr = {rate:8.4f}')
        print("    tp", round(float(model_Akw.tp), 3))
        print("    tpp", round(float(model_Akw.tpp), 3))
        print("    mu", round(float(model_Akw.mu), 3))
        print("    eta", round(float(model_Akw.eta), 3))

if args.plot:
    print_fkw(model_Akw)
if args.plot_loss:
    plt.plot(loss_history)
    plt.show()

print('real self')
print_fkw(model_Akw.neural_self, idx=0)
print('imaginary self')
print_fkw(model_Akw.neural_self, idx=1)

if args.animate:
    print("preparing animation")
    kx = torch.linspace(0,np.pi,101)
    ky = torch.linspace(0,np.pi,101)
    kxx, kyy = torch.meshgrid(kx, ky)
    Akw_on_mesh = model_history[0](kxx,kyy,0)
    error_on_mesh = torch.abs( model_history[0](kxx,kyy,0) - target_Akw(kxx,kyy,0))

    fig = plt.figure(figsize=(4, 6), dpi=80, facecolor='w', edgecolor='k')
    ax1 = plt.subplot(211)
    ax1.set_aspect('equal', 'box')
    ax1.pcolormesh(
        np.transpose(kx),
        np.transpose(ky),
        np.transpose(Akw_on_mesh.detach()),
    )
    
    ax2 = plt.subplot(212)
    ax2.set_aspect('equal', 'box')
    error = ax2.pcolormesh(
        np.transpose(kx),
        np.transpose(ky),
        np.transpose(error_on_mesh.detach()),
        vmin=0, vmax = 5,
    )
    fig.colorbar(error, ax=ax2)

    def animate(i):
        new_Akw_on_mesh = model_history[i](kxx,kyy,0)
        new_error_on_mesh = torch.abs(model_history[i](kxx,kyy,0) - target_Akw(kxx,kyy,0))
        ax1.pcolormesh(
            np.transpose(kx),
            np.transpose(ky),
            np.transpose(new_Akw_on_mesh.detach()),
        )
        ax2.pcolormesh(
            np.transpose(kx),
            np.transpose(ky),
            np.transpose(new_error_on_mesh.detach()),
        )

    anim = FuncAnimation(fig, animate, 
        frames=len(model_history), 
        interval=100, 
        blit=False
    )
    anim.save('anim.gif')
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
    'lr' : 0.01,
    'loss' : 'KL',
    'schedule' : '',
    'factor' : 0.9,
    'plot' : False,
    'plot_loss' : False,
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


def spectral_weight(omega, energy, eta=0.05):
    return eta / ((omega - energy)**2 + eta**2)


def print_fkw(function_of_kx_ky_w, w=0, save_path=None):
    kx = torch.linspace(0,np.pi,101)
    ky = torch.linspace(0,np.pi,101)
    kxx, kyy = torch.meshgrid(kx, ky)
    Akw_on_mesh = function_of_kx_ky_w(kxx,kyy,w)

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


target_Akw = lambda kx,ky,w: spectral_weight(w, dispersion(kx,ky))
print('target')
# print_fkw(target_Akw)


# differentiable spectral weight

class DifferentiableSW(nn.Module):
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


model_Akw = DifferentiableSW()
print('starting model')
# print_fkw(model_Akw)


def manual_KL(Amodel, Atarget):
    return (Atarget*Atarget.log() - Atarget*Amodel.log()).mean()

def my_loss(Amodel, Atarget):
    return (Atarget*Atarget.log() - Atarget*Amodel.log()).mean()



if args.optim == 'adam' or args.optim == 'Adam':
    optimizer = torch.optim.Adam(model_Akw.parameters(), lr=args.lr)
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
        loss = my_loss(results,targets)
    elif args.loss == 'mse' or args.loss == 'MSE':
        loss = F.l1_loss(results, targets)
    else :
        loss = F.kl_div(results.log(), targets)
    loss.backward()

    if args.schedule == 'KL':
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
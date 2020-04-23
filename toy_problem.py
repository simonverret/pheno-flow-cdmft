# Copyright (c) Simon Verret, 2020 
#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy


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
print_fkw(target_Akw)


#%% differentiable spectral weight

class DifferentiableSW(nn.Module):
    def __init__(self, tp0=-0.4, tpp0=0.1, mu0=-1, eta0=1.0):
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
print_fkw(model_Akw)


#%% fitting one batch

loss_func = torch.nn.KLDivLoss()
optimizer = torch.optim.Adam(model_Akw.parameters(), lr=0.01)

model_history = []
for batch in range(1,300):
    # sample a thousand random (kx,ky,w) points
    batch_size = 1000
    kx_batch = torch.FloatTensor(batch_size).uniform_(0, np.pi)
    kx_batch = torch.FloatTensor(batch_size).uniform_(0, np.pi)
    w_batch = torch.FloatTensor(batch_size).uniform_(-2, 2)

    optimizer.zero_grad()
    results = model_Akw(kx_batch,kx_batch,w_batch)
    targets = target_Akw(kx_batch,kx_batch,w_batch)
    loss = loss_func(results,targets)
    loss.backward()
    optimizer.step()

    if batch%10 == 1:
        model_history.append(deepcopy(model_Akw))
        print("tp", round(float(model_Akw.tp), 3))
        print("tpp", round(float(model_Akw.tpp), 3))
        print("mu", round(float(model_Akw.mu), 3))
        print("eta", round(float(model_Akw.eta), 3))
        # print_fkw(model_Akw)

#%%
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(4, 6), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(111)
ax.set_aspect('equal', 'box')
kx = torch.linspace(0,np.pi,101)
ky = torch.linspace(0,np.pi,101)
kxx, kyy = torch.meshgrid(kx, ky)
Akw_on_mesh = model_history[0](kxx,kyy,0)
heatmap = ax.pcolormesh(
    np.transpose(kx),
    np.transpose(ky),
    np.transpose(Akw_on_mesh.detach()),
)

print(heatmap)


#%%
def animate(i):
    Akw_on_mesh = model_history[i](kxx,kyy,0)
    return ax.pcolormesh(
        np.transpose(kx),
        np.transpose(ky),
        np.transpose(Akw_on_mesh.detach()),
    )


anim = FuncAnimation(fig, animate, 
    frames=len(model_history), 
    interval=100, 
    blit=False
)
anim.save('anim.gif')
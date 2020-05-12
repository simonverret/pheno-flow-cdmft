import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def print_spectrum(function_of_kx_ky_w, save_path=None, idx=0):
    kx = torch.cat((
        torch.linspace(0,np.pi,101).float(),
        np.pi*torch.ones(101).float(),
        torch.linspace(np.pi,0,143).float(),
    ))
    ky = torch.cat((
        torch.zeros(101).float(),
        torch.linspace(0,np.pi,101).float(),
        torch.linspace(np.pi,0,143).float(),
    ))
    w = torch.linspace(-5,5,100)
    kxx, ww = torch.meshgrid(kx, w)
    kyy, ww = torch.meshgrid(ky, w)

    Akw_on_mesh = function_of_kx_ky_w(kxx,kyy,ww)
    if isinstance(Akw_on_mesh, tuple):
        Akw_on_mesh = Akw_on_mesh[idx]

    Akw_on_mesh = Akw_on_mesh.squeeze()
    fig = plt.figure(figsize=(4, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)
    ax.set_aspect('equal', 'box')
    ax.pcolormesh(
        np.transpose(Akw_on_mesh.detach()),
    )
    if save_path is not None: 
        plt.savefig(save_path)
    plt.show()


def print_fermi_surface(function_of_kx_ky_w, w=0, save_path=None, idx=0):
    kx = torch.linspace(0,np.pi,101).float()
    ky = torch.linspace(0,np.pi,101).float()
    kxx, kyy = torch.meshgrid(kx, ky)

    ## for compatibility with neural nets
    kxx = kxx.unsqueeze(-1)
    kyy = kyy.unsqueeze(-1)
    w = w*torch.ones_like(kxx).float()
    
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

def animate_fermi_surface(model_history, target_model):
    print("preparing animation")
    kx = torch.linspace(0,np.pi,101)
    ky = torch.linspace(0,np.pi,101)
    kxx, kyy = torch.meshgrid(kx, ky)
    Akw_on_mesh = model_history[0](kxx,kyy,0)
    target_Akw_on_mesh = target_model.spectral_weight(kxx,kyy,0)

    error_on_mesh = torch.abs(Akw_on_mesh - target_Akw_on_mesh)

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
        new_error_on_mesh = torch.abs(new_Akw_on_mesh - target_Akw_on_mesh)
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
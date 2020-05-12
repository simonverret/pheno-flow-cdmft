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
    ax.pcolormesh(np.transpose(Akw_on_mesh.detach()))
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

def animation(model_history, target_model):
    print("preparing animation")
    
    # fermi surface 
    kx = torch.linspace(0,np.pi,101)
    ky = torch.linspace(0,np.pi,101)
    kxx, kyy = torch.meshgrid(kx, ky)
    Akw_on_mesh = model_history[0](kxx,kyy,0)
    target_Akw_on_mesh = target_model.spectral_weight(kxx,kyy,0)

    fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    ax1 = plt.subplot(221)
    ax1.set_aspect('equal', 'box')
    ax1.pcolormesh(
        np.transpose(kx),
        np.transpose(ky),
        np.transpose(Akw_on_mesh.detach()),
    )
    
    ax2 = plt.subplot(223)
    ax2.set_aspect('equal', 'box')
    error = ax2.pcolormesh(
        np.transpose(kx),
        np.transpose(ky),
        np.transpose(target_Akw_on_mesh.detach()),
        vmin=0, vmax = 5,
    )
    fig.colorbar(error, ax=ax2)


    # spectrum
    skx = torch.cat((
        torch.linspace(0,np.pi,101).float(),
        np.pi*torch.ones(101).float(),
        torch.linspace(np.pi,0,143).float(),
    ))
    sky = torch.cat((
        torch.zeros(101).float(),
        torch.linspace(0,np.pi,101).float(),
        torch.linspace(np.pi,0,143).float(),
    ))
    sw = torch.linspace(-5,5,100)
    skxx, sww = torch.meshgrid(skx, sw)
    skyy, sww = torch.meshgrid(sky, sw)
    spectrum_on_mesh = model_history[0](skxx,skyy,sww)
    target_spectrum_on_mesh = target_model.spectral_weight(skxx,skyy,sww)
    ax3 = plt.subplot(222)
    ax3.pcolormesh(np.transpose(spectrum_on_mesh.detach()))
    
    ax4 = plt.subplot(224)
    ax4.pcolormesh(np.transpose(target_spectrum_on_mesh.detach()))

    def animate(i):
        new_Akw_on_mesh = model_history[i](kxx,kyy,0)
        new_spectrum_on_mesh = model_history[i](skxx,skyy,sww)
        ax1.pcolormesh(
            np.transpose(kx),
            np.transpose(ky),
            np.transpose(new_Akw_on_mesh.detach()),
        )
        ax2.pcolormesh(
            np.transpose(kx),
            np.transpose(ky),
            np.transpose(target_Akw_on_mesh.detach()),
        )
        ax3.pcolormesh(np.transpose(new_spectrum_on_mesh.detach()))
        ax4.pcolormesh(np.transpose(target_spectrum_on_mesh.detach()))

    anim = FuncAnimation(fig, animate, 
        frames=len(model_history), 
        interval=100, 
        blit=False
    )
    anim.save('anim.gif')
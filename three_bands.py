#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

class Three_band_model():
    def __init__(self, charge_tansfer=-2.61, tpd=1.39, tpp=0.64, tpp2=0.103):
        self.charge_tansfer = charge_tansfer
        self.tpd = tpd
        self.tpp = tpp
        self.tpp2 = tpp2

    def hamiltonian_matrix(self, kx, ky, ed=0):
        ed = ed * torch.ones_like(kx)
        ep = ed - self.charge_tansfer
        
        H11 = ed
        H12 = 2*self.tpd*torch.sin(kx/2)
        H13 = -2*self.tpd*torch.sin(ky/2)
        H21 = H12
        H22 = ep + self.tpp2*torch.cos(kx)
        H23 = -4*self.tpp*torch.sin(kx/2)*torch.sin(ky/2)
        H31 = H13
        H32 = H23
        H33 = ep + 2*self.tpp2*torch.cos(ky)

        L1 = torch.stack((H11, H12, H13), dim=-1)
        L2 = torch.stack((H21, H22, H23), dim=-1)
        L3 = torch.stack((H31, H32, H33), dim=-1)
        H = torch.stack((L1, L2, L3), dim=-1)
        return H
    
    def bands(self, kx, ky, ed=0):
        H = self.hamiltonian_matrix(kx, ky, ed)
        eigenvals = torch.symeig(H)
        return eigenvals
        

if __name__=="__main__":
        
    three_bands_model = Three_band_model()
    kx = torch.linspace(0,np.pi,101).float()
    ky = torch.linspace(0,np.pi,101).float()
    kxx, kyy = torch.meshgrid(kx, ky)
    energies = three_bands_model.bands(kx,ky, ed=-1.)

    plt.plot(kx, energies[0])
    plt.plot(kx, energies[0])
    plt.plot(kx, energies[0])
    plt.show()
import torch
from math import pi
import matplotlib.pyplot as plt

class One_band():
    def __init__(self, mu=-1, t=1, tp=-0.3, tpp=0.2):
        self.t = t
        self.mu = mu
        self.tp = tp
        self.tpp = tpp

    def dispersion(self, kx, ky):
        coskx = torch.cos(kx)
        cosky = torch.cos(ky)
        cos2kx = torch.cos(2*kx)
        cos2ky = torch.cos(2*ky)
        
        xi = -2*self.t*(coskx + cosky)
        xi -= 4*self.tp*coskx*cosky 
        xi -= 2*self.tpp*(cos2kx + cos2ky) 
        xi -= self.mu 
        return xi

    def spectral_weight(self, kx, ky, w, eta=0.1):
        xi = self.dispersion(kx,ky)
        return eta / ((w - xi)**2 + eta**2)


class YRZ_model(One_band):
    def __init__(self, delta=0.3, **kw):
        super().__init__(**kw)
        self.delta = 0.3

    def xi0(self, kx, ky):
        return -2*self.t*(torch.cos(kx) + torch.cos(ky))

    def self_energy(self, kx, ky, omega, delta=0.2, eta=0.05):
        xi0 = self.xi0(kx, ky)
        re_sigma = delta**2 * (omega - xi0) / ((omega - xi0)**2 + eta**2)
        im_sigma = delta**2 * eta / ((omega - xi0)**2 + eta**2)
        return re_sigma, im_sigma

    def spectral_weight(self, kx, ky, w, eta=0.1):
        re_sigma, im_sigma = self.self_energy(kx,ky,w, self.delta, eta)
        xi = self.dispersion(kx,ky)
        return eta/((w - xi + re_sigma)**2 + (eta + im_sigma)**2)


class Three_bands():
    def __init__(self, charge_tansfer=-2.61, tpd=1.39, tpp=0.64, tpp2=0.103):
        self.charge_tansfer = charge_tansfer
        self.tpd = tpd
        self.tpp = tpp
        self.tpp2 = tpp2

    def hamiltonian_matrix(self, kx, ky, ed=0):
        ed = ed * torch.ones_like(kx)
        ep = ed - self.charge_tansfer
        
        # element-wise definition of the hamiltonian
        H11 = ed
        H12 = 2*self.tpd*torch.sin(kx/2)
        H13 = -2*self.tpd*torch.sin(ky/2)
        H21 = H12
        H22 = ep + self.tpp2*torch.cos(kx)
        H23 = -4*self.tpp*torch.sin(kx/2)*torch.sin(ky/2)
        H31 = H13
        H32 = H23
        H33 = ep + 2*self.tpp2*torch.cos(ky)

        # cannot make a tensor from a list of tensor so let's stack them up
        row1 = torch.stack((H11, H12, H13), dim=-1)
        row2 = torch.stack((H21, H22, H23), dim=-1)
        row3 = torch.stack((H31, H32, H33), dim=-1)
        H = torch.stack((row1, row2, row3), dim=-1)
        return H
    
    def bands(self, kx, ky, ed=0):
        H = self.hamiltonian_matrix(kx, ky, ed)
        eigenvals = torch.symeig(H)
        return eigenvals

    def spectral_weight(self, kx, ky, ww, eta=0.05, ed=-1.8):
        xi = self.bands(kx,ky, ed)[0]
        # unsqueeze to vectorize computation on the three bands
        if not isinstance(ww, torch.FloatTensor):
            ww = torch.tensor([ww])
        kx = kx.unsqueeze(-1)
        ky = ky.unsqueeze(-1)
        ww = ww.unsqueeze(-1)
        # sum the three spectral weights
        return (eta/((ww - xi)**2 + eta**2)).sum(-1)


def main():
    three_bands_model = Three_band_model()

    kx = torch.cat((
        torch.linspace(0,pi,101).float(),
        pi*torch.ones(101).float(),
        torch.linspace(pi,0,143).float(),
    ))
    ky = torch.cat((
        torch.zeros(101).float(),
        torch.linspace(0,pi,101).float(),
        torch.linspace(pi,0,143).float(),
    ))
    energies = three_bands_model.bands(kx,ky, ed=-1.)

    plt.plot(energies[0])
    plt.plot(energies[0])
    plt.plot(energies[0])
    plt.show()


if __name__=="__main__":
    main()        
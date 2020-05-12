# Copyright (c) Simon Verret, 2020 
#%%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from copy import deepcopy
import argparse

from band_models import One_band, Three_bands
from differentiable_models import Differentiable_one_band
from plot_utils import (
    print_spectrum, 
    print_fermi_surface, 
    animate_fermi_surface
)

default_args = {
    'lr' : 0.05,
    'loss' : 'JS',
    'schedule' : '',
    'factor' : 0.9,
    'plot' : True,
    'plot_loss' : True,
    'plot_self' : False,
    'animate' : True,
    'optim' : 'Adam',
    'batch_size' : 1000,
    'epochs': 500,
    'window': 0.5,
}

parser = argparse.ArgumentParser()
for key, val in default_args.items():
    if type(val) is list:
        parser.add_argument("--"+key, nargs="+", type=type(val[0]), default=val)
    elif type(val) is bool and val is False:
        parser.add_argument("--"+key, action="store_true", default=val)
    elif type(val) is bool and val is True:
        parser.add_argument("--no_"+key, dest=key, action="store_false")
    elif val is None:
        parser.add_argument("--"+key, default=val)
    else:
        parser.add_argument("--"+key, type=type(val), default=val)
args = parser.parse_known_args()[0]

target_model = Three_bands(charge_tansfer=-2.61, tpd=1.29, tpp=0.64, tpp2=0.103)
# print('target')
# print_spectrum(target_model.spectral_weight)

model = Differentiable_one_band()
print('starting model')
print_fermi_surface(model.spectral_weight)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
elif args.optim == 'sgd' or args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


print('training')
model_history = []
loss_history = []
for epoch in range(1,args.epochs):
    kx_batch = torch.FloatTensor(args.batch_size).uniform_(-np.pi, np.pi)
    ky_batch = torch.FloatTensor(args.batch_size).uniform_(-np.pi, np.pi)
    W = args.window/2
    w_batch = torch.FloatTensor(args.batch_size).uniform_(-W, W)

    optimizer.zero_grad()
    results = model(kx_batch,ky_batch,w_batch)
    targets = target_model.spectral_weight(kx_batch,ky_batch,w_batch)

    # I checked this:
    # assert kl_div(results,targets) -  F.kl_div(results.log(),targets) < 1e-4
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
    if epoch%(args.epochs//60) == 1:
        model_history.append(deepcopy(model))
        print(f'  loss = {loss:8.4f}    lr = {rate:8.4f}')
        print("    tp", round(float(model.tp), 3))
        print("    tpp", round(float(model.tpp), 3))
        print("    mu", round(float(model.mu), 3))
        print("    eta", round(abs(float(model.eta)), 3))

if args.plot:
    print_fermi_surface(model.spectral_weight)
if args.plot_loss:
    plt.plot(loss_history)
    plt.show()
if args.animate:
    animate_fermi_surface(model_history, target_model)
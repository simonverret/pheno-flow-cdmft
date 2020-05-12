#
# Differentiable phenological models
# Copyright (c) Simon Verret, 2020 
#

import sys
from numpy import pi, ceil
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from copy import deepcopy
import argparse

from band_models import One_band, Three_bands
from differentiable_models import Differentiable_one_band
from plot_utils import print_spectrum, print_fermi_surface, animation


def kl_div(Amodel, Atarget):
    """Kullback-Leibler Divergence"""
    return (Atarget*Atarget.log() - Atarget*Amodel.log()).mean()


def js_div(Amodel, Atarget):
    """Jensen-Shannone Divergence"""
    Amiddle = (Amodel + Atarget)/2
    return (kl_div(Amodel, Amiddle) + kl_div(Atarget, Amiddle))/2


def manual_loss(Amodel, Atarget):
    return (Atarget*Atarget.log() - Atarget*Amodel.log() - Amodel*Atarget.log()).mean()


default_args = {
    'lr' : 0.1,
    'loss' : 'KL',
    'schedule' : 'exp',
    'factor' : 0.998,
    'plot' : True,
    'plot_loss' : False,
    'animate' : False,
    'optim' : 'SGD',
    'batch_size' : 500,
    'batch_schedule': 1.001,
    'max_batch_size': 1000,
    'epochs': 5000,
    'window': 3.0
}

help = {
    'lr' : "learning rate",
    'loss' : "divergence or error to optimize: KL, JS, MAE or MSE",
    'schedule' : "learning rate decay schedule type: exp or loss",
    'factor' : "factor for exponential decay of learning rate",
    'plot' : "will disaply plots of spectral weights during the fit",
    'plot_loss' : "will display a plot of the learning curve",
    'animate' : "will produce an animation of the learning at the end",
    'optim' : "select the optimizer: Adam or SGD",
    'batch_size' : "size of the random sample used for fit",
    'batch_schedule': "growth factor of the batch with each epoch",
    'max_batch_size': "maximum size of random batch when batch schedule",
    'epochs': "number of epoch (1 batch per epoch)",
    'window': "size of the energy window from which to consider samples"
}

parser = argparse.ArgumentParser()
for key, val in default_args.items():
    if type(val) is list:
        parser.add_argument("--"+key, nargs="+", type=type(val[0]), default=val, help=help[key])
    elif type(val) is bool and val is False:
        parser.add_argument("--"+key, action="store_true", default=val, help=help[key])
    elif type(val) is bool and val is True:
        parser.add_argument("--no_"+key, dest=key, action="store_false", help=help[key])
    elif val is None:
        parser.add_argument("--"+key, default=val, help=help[key])
    else:
        parser.add_argument("--"+key, type=type(val), default=val, help=help[key])
args = parser.parse_known_args()[0]


target_model = Three_bands(ed=-2.5, ep=-4.0, tpd=1.5, tpp1=1.0, tpp2=0.2)
# target_model = One_band()
if args.plot:
    print('target')
    print_spectrum(target_model.spectral_weight)

model = Differentiable_one_band(tp=0, tpp=0, mu=0, eta=0.2)
if args.plot:
    print('starting model')
    print_spectrum(model.spectral_weight)

if args.optim == 'adam' or args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
elif args.optim == 'sgd' or args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


print('training')
model_history = []
loss_history = []
rate = args.lr

float_batch_size = float(args.batch_size)
batch_size = int(float_batch_size) 

for epoch in range(1,args.epochs):

    if batch_size < args.max_batch_size:
        float_batch_size = args.batch_schedule*float_batch_size
    batch_size = int(float_batch_size) 
    
    kx_batch = torch.FloatTensor(batch_size).uniform_(-pi, pi)
    ky_batch = torch.FloatTensor(batch_size).uniform_(-pi, pi)
    W = args.window/2
    w_batch = torch.FloatTensor(batch_size).uniform_(-W, W)
    
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
    elif args.loss in ['KL', 'kl', 'Kullback-Leibler'] :
        loss = kl_div(results, targets)
    loss.backward()

    if args.schedule in ['loss','KL']:
        rate = args.lr * loss.item()
        for g in optimizer.param_groups: g['lr'] = rate
    elif args.schedule in ['exp']:
        rate = args.lr * (args.factor)**(epoch)
        for g in optimizer.param_groups: g['lr'] = rate
    else:
        rate = args.lr
    loss_history.append(loss.item())

    optimizer.step()
    if epoch%(ceil(args.epochs/100)) == 0:
        model_history.append(deepcopy(model))
        print(f"batch {epoch}")
        print(f'  loss = {loss:8.4f}   lr = {rate:8.8f}   bs = {batch_size}')
        print("    t = ", round(float(model.t), 3))
        print("    tp = ", round(float(model.tp), 3))
        print("    tpp = ", round(float(model.tpp), 3))
        print("    mu = ", round(float(model.mu), 3))
        print("    eta = ", round(abs(float(model.eta)), 3))
    

if args.plot:
    print_spectrum(model.spectral_weight)
if args.plot_loss:
    plt.plot(loss_history)
    plt.show()
if args.animate:
    animation(model_history, target_model)


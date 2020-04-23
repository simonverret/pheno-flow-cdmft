#!/usr/local/bin/python3

# zlib license:

# Copyright (c) 2019 Maxime Charlebois

# This software is provided 'as-is', without any express or implied
# warranty. In no event will the authors be held liable for any damages
# arising from the use of this software.

# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:

# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#   misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.


# This is an altered version
# Copyright (c) Simon Verret, 2020

# As required by the above Lisence, here is a note to mention that this code
# have been modified to produce a PDF figure illustrating the periodization
# results.



#%%

import numpy as np
from scipy import linalg as LA
from copy  import deepcopy
import sys, os, re
from scipy.linalg import eigh,eig,eigvals,eigvalsh

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)

t=1.
tp=-0.3
tpp=0.2

Nkx = 101
mu=-0.3
M=0.4
omega = 0
eta = 0.08

# t=1.
# tp=-0.4
# tpp=0.

# Nkx = 41
# mu=-0.4
# M=0.5
# omega = 0.
# eta = 0.1

#mu=0.2
#M=0.6

z = omega + 1.j*eta + mu
Ry = np.array([0,0,1,1])
Rx = np.array([0,1,1,0])

tc  = np.array([[ 0., -t , -tp, -t ],
                [-t ,  0., -t , -tp],
                [-tp, -t ,  0., -t ],
                [-t , -tp, -t ,  0.]])

tc2 = np.array([[ 0.,  t , -tp,  t ],
                [ t ,  0.,  t , -tp],
                [-tp,  t ,  0.,  t ],
                [ t , -tp,  t ,  0.]])


Nky = Nkx
Akw = np.zeros([Nkx,Nky])
AkwM = np.zeros([Nkx,Nky])
AkwCT = np.zeros([Nkx,Nky])
AkwExact = np.zeros([Nkx,Nky])
tkQ= np.zeros([4,4],dtype='complex')

for xx in range(Nkx):
  # print xx, ' of ', Nkx
  for yy in range(Nky):
    kx = 1.*np.pi*1.*float(xx)/(float(Nkx-1.))
    ky = 1.*np.pi*1.*float(yy)/(float(Nky-1.))
    
    ex  = np.exp(-2.j*kx)
    emx = np.exp( 2.j*kx)
    ey  = np.exp(-2.j*ky)
    emy = np.exp( 2.j*ky)
    
    dtk = np.array([[-tpp*(emx+ex+ey+emy)     ,  -t*ex                  ,  -tp*(ex + ey + ex*ey),  -t*ey                   ],
                    [-t*emx                   ,  -tpp*(emx+ex+ey+emy)   ,  -t*ey                ,  -tp*(emx + ey + emx*ey) ],
                    [-tp*(emx + emy + emx*emy),  -t*emy                 ,  -tpp*(emx+ex+ey+emy) ,  -t*emx                  ],
                    [-t*emy                   ,  -tp*(ex + emy + ex*emy),  -t*ex                ,  -tpp*(emx+ex+ey+emy)    ]])

    
    dtk2= np.array([[-tpp*(emx+ex+ey+emy)     ,   t*ex                  ,  -tp*(ex + ey + ex*ey),   t*ey                   ],
                    [ t*emx                   ,  -tpp*(emx+ex+ey+emy)   ,   t*ey                ,  -tp*(emx + ey + emx*ey) ],
                    [-tp*(emx + emy + emx*emy),   t*emy                 ,  -tpp*(emx+ex+ey+emy) ,   t*emx                  ],
                    [ t*emy                   ,  -tp*(ex + emy + ex*emy),   t*ex                ,  -tpp*(emx+ex+ey+emy)    ]])
    
    tpp_c=0
    dtk3= np.array([[-tpp_c*(emx+ex+ey+emy)     ,   t*ex                  ,  -tp*(ex + ey + ex*ey),   t*ey                   ],
                    [ t*emx                   ,  -tpp_c*(emx+ex+ey+emy)   ,   t*ey                ,  -tp*(emx + ey + emx*ey) ],
                    [-tp*(emx + emy + emx*emy),   t*emy                 ,  -tpp_c*(emx+ex+ey+emy) ,   t*emx                  ],
                    [ t*emy                   ,  -tp*(ex + emy + ex*emy),   t*ex                ,  -tpp_c*(emx+ex+ey+emy)    ]])

    
    cumul = LA.inv( z*np.identity(4) - M*M*LA.inv(z*np.identity(4) - tc2) )
    denom = z*np.identity(4) -tc - dtk - M*M*LA.inv(z*np.identity(4) - tc2)
    denomCT = z*np.identity(4) -tc - dtk - M*M*LA.inv(z*np.identity(4) - tc2 - dtk3)
    denomExact = z*np.identity(4) -tc - dtk - M*M*LA.inv(z*np.identity(4) - tc2 - dtk2)

    G_ktilde = LA.inv(denom)
    G_ktilde_CT = LA.inv(denomCT)
    G_ktilde_exact = LA.inv(denomExact)

    G_per = 0.0
    cumul_per = 0.0
    denom_per = 0.0
    G_CT = 0.0
    G_exact = 0.0

    for jj in range(0,4):
      for ii in range(0,4):
        G_per += (1./4.) * G_ktilde[ii,jj]  * np.exp(1.0j*(((Rx[jj]-Rx[ii]))*kx + ((Ry[jj]-Ry[ii]))*ky) )
        # denom_per += (1./4.) * denom[ii,jj] * np.exp(1.0j*(((Rx[jj]-Rx[ii]))*kx + ((Ry[jj]-Ry[ii]))*ky) )
        cumul_per += (1./4.) * cumul[ii,jj]  * np.exp(1.0j*(((Rx[jj]-Rx[ii]))*kx + ((Ry[jj]-Ry[ii]))*ky) )
        G_CT += (1./4.) * G_ktilde_CT[ii,jj]  * np.exp(1.0j*(((Rx[jj]-Rx[ii]))*kx + ((Ry[jj]-Ry[ii]))*ky) )
        G_exact += (1./4.) * G_ktilde_exact[ii,jj]  * np.exp(1.0j*(((Rx[jj]-Rx[ii]))*kx + ((Ry[jj]-Ry[ii]))*ky) )
        

    #G_per = 1./denom_per 
    Akw[xx,yy] = -np.imag(G_per)

    epsilon_k = -2*t*(np.cos(kx)+np.cos(ky)) - 4*tp*np.cos(kx)*np.cos(ky)
    AkwM[xx,yy] = -2*np.imag(1/(1/cumul_per - epsilon_k))

    AkwCT[xx,yy] = -np.imag(G_CT)
    AkwExact[xx,yy] = -np.imag(G_exact)


#%%
X,Y = np.meshgrid(np.array(range(Nkx)),np.array(range(Nky)))

plt.rc('text', usetex=True)
plt.rc('font', family='Serif')
plt.rc('text.latex', preamble=r'''
  \usepackage[utf8]{inputenc}
  \usepackage{amssymb}
  \usepackage{amsmath}
  \usepackage{esint}
  \usepackage{amsfonts}
  \usepackage{dsfont}
  \usepackage{bm}
  \renewcommand{\vec}[1]{\bm{\mathrm{#1}}}
''')
  # won't work in vscode ipython notebook
  # \usepackage{txfonts}

w = 3.5
h = 2.5
fig = plt.figure(figsize=(w, h), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['xtick.major.size'] = 2

ax4 = plt.subplot(234)
ax5 = plt.subplot(235, sharey=ax4)
ax6 = plt.subplot(236, sharey=ax4)
ax1 = plt.subplot(231, sharex=ax4)
ax2 = plt.subplot(232, sharey=ax1, sharex=ax5)
ax3 = plt.subplot(233, sharey=ax1, sharex=ax6)

# plt.subplots_adjust(wspace=0.1, hspace=0.03)

ax1.set_yticks([0, (Nkx-1)/2 ,Nkx-1])
ax1.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$" ,r"$\pi$"])
ax4.set_yticks([0, (Nkx-1)/2 , Nkx-1])
ax4.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$" ,r"$\pi$"])
ax4.set_xticks([0, (Nkx-1)/2 , Nkx-1])
ax4.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$" ,r"$\pi$"])
ax5.set_xticks([0, (Nkx-1)/2 , Nkx-1])
ax5.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$" ,r"$\pi$"])
ax6.set_xticks([0, (Nkx-1)/2 , Nkx-1])
ax6.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$" ,r"$\pi$"])


mykwargs = {'cmap':cm.bone_r, 'rasterized':True, 'antialiased':False}

ax1.set_aspect('equal', 'box')
plot1 = ax1.pcolormesh(np.transpose(Y),np.transpose(X),np.transpose(AkwM),**mykwargs)
ax1.set_ylabel(r"$k_y$")
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.text(0.75*(Nkx-1), 0.8*(Nkx-1), r"(a)")
ax1.text(0.05*(Nkx-1), 0.22*(Nkx-1), r"\footnotesize{c-AF}")
ax1.text(0.05*(Nkx-1), 0.07*(Nkx-1), r"\footnotesize{M-scheme}")
# fig.colorbar(plot1, ax=ax1)


ax2.set_aspect('equal', 'box')
ax2.pcolormesh(np.transpose(Y),np.transpose(X),np.transpose(Akw),**mykwargs)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
ax2.text(0.75*(Nkx-1), 0.8*(Nkx-1), r"(b)")
ax2.text(0.05*(Nkx-1), 0.22*(Nkx-1), r"\footnotesize{c-AF}")
ax2.text(0.05*(Nkx-1), 0.07*(Nkx-1), r"\footnotesize{G-scheme}")

ax3.set_aspect('equal', 'box')
ax3.pcolormesh(np.transpose(Y),np.transpose(X),np.transpose(AkwCT),**mykwargs)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.text(0.75*(Nkx-1), 0.8*(Nkx-1), r"(c)")
ax3.text(0.05*(Nkx-1), 0.22*(Nkx-1), r"\footnotesize{c-AF}")
ax3.text(0.05*(Nkx-1), 0.07*(Nkx-1), r"\footnotesize{Compact}")


ax4.set_aspect('equal', 'box')
ax4.pcolormesh(np.transpose(Y),np.transpose(X),np.transpose(0.1*AkwM),**mykwargs)
ax4.set_ylabel(r"$k_y$")
ax4.set_xlabel(r"$k_x$")
ax4.text(0.75*(Nkx-1), 0.8*(Nkx-1), r"(d)")
ax4.text(0.05*(Nkx-1), 0.22*(Nkx-1), r"\footnotesize{CDMFT}")
ax4.text(0.05*(Nkx-1), 0.07*(Nkx-1), r"\footnotesize{M-scheme}")

ax5.set_aspect('equal', 'box')
ax5.pcolormesh(np.transpose(Y),np.transpose(X),np.transpose(0.1*Akw),**mykwargs)
ax5.set_xlabel(r"$k_x$")
plt.setp(ax5.get_yticklabels(), visible=False)
ax5.text(0.75*(Nkx-1), 0.8*(Nkx-1), r"(e)")
ax5.text(0.05*(Nkx-1), 0.22*(Nkx-1), r"\footnotesize{CDMFT}")
ax5.text(0.05*(Nkx-1), 0.07*(Nkx-1), r"\footnotesize{G-scheme}")

ax6.set_aspect('equal', 'box')
ax6.pcolormesh(np.transpose(Y),np.transpose(X),np.transpose(AkwExact),**mykwargs)
ax6.set_xlabel(r"$k_x$")
plt.setp(ax6.get_yticklabels(), visible=False)
ax6.text(0.75*(Nkx-1), 0.8*(Nkx-1), r"(f)")
ax6.text(0.05*(Nkx-1), 0.22*(Nkx-1), r"\footnotesize{AF}")
ax6.text(0.05*(Nkx-1), 0.07*(Nkx-1), r"\footnotesize{Exact}")

plt.tight_layout()
plt.savefig("figure"+".pdf",dpi=300)
# plt.show()



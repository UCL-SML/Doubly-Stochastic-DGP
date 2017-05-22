#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:37:34 2017

@author: hughsalimbeni
"""

import sys
sys.path.append('../src')
sys.path.append('../../DNSGP/GPflow')

import numpy as np
import tensorflow as tf

from GPflow.likelihoods import Gaussian
from GPflow.kernels import RBF

from dgp import DGP

import matplotlib.pyplot as plt


Ns = 300
Xs = np.linspace(0, 1, Ns)[:, None]

L = 2
kernels = []
for l in range(L):
    kernels.append(RBF(1, lengthscales=0.2, variance=1))
    
kernels = [RBF(1, lengthscales=0.2, variance=1), RBF(2, lengthscales=0.2, variance=1)]
    
    
    

N, M = 50, 25
X = np.random.uniform(0, 1, N)[:, None]
Z = np.random.uniform(0, 1, M)[:, None]
f = lambda x: 0. if x<0.5 else 1.
Y = np.reshape([f(x) for x in X], X.shape) + np.random.randn(*X.shape)*1e-2





m_dgp = DGP(X, Y, Z, kernels, Gaussian(), num_samples=1)
for layer in m_dgp.layers[:-1]:
    layer.q_sqrt = layer.q_sqrt.value * 1e-5

class CB(object):
    def __init__(self, model, record_every=10):
        self.model = model
        self.i = 0
        self.res = []
        self.record_every = record_every
    def cb(self, x):
        self.i += 1
        if self. i % self.record_every == 0:
            self.model.set_state(x)
            self.res.append(self.model.compute_log_likelihood())
        
cb = CB(m_dgp)
m_dgp.optimize(tf.train.AdamOptimizer(0.01), maxiter=10000, callback=cb.cb)

plt.plot(cb.res)
plt.show()


f, axs = plt.subplots(L, 1, figsize=(4, 2*L), sharex=True)
if L == 1:
    axs = [axs, ]

for _ in range(10):
    samples = m_dgp.predict_all_layers_full_cov(Xs)
    for s, ax in zip(samples, axs):
        S, N, D  = s.shape
        for d in range(D):
            ax.plot(Xs.flatten(), s[:, :, d].T, color='r', alpha=0.2)
    

axs[-1].scatter(X, Y)
plt.show()














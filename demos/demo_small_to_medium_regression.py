# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:36:13 2017

@author: hughsalimbeni
"""

import sys
sys.path.append('../src')
sys.path.append('../../DNSGP/GPflow')

import numpy as np
import tensorflow as tf

from GPflow.likelihoods import Gaussian
from GPflow.kernels import RBF
from GPflow.mean_functions import Constant
from GPflow.sgpr import SGPR

from dgp import DGP

import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.misc import logsumexp
from scipy.stats import norm
from get_data import get_regression_data


X, Y, Xs, Ys = get_regression_data('energy', 0)

Y_mean, Y_std = np.average(Y), np.std(Y)

M = 100
Z = kmeans2(X, M, minit='points')[0] 

D = X.shape[1]


def make_dgp(L):
    kernels = []
    for l in range(L):
        kernels.append(RBF(D, lengthscales=1., variance=1.))
    model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=1)

    # we haven't whitened the outputs, so set some sensible values
    model.layers[-1].kern.variance = Y_std**2
    model.likelihood.variance = Y_std*0.1 
    model.layers[-1].mean_function = Constant(Y_mean)
    model.layers[-1].mean_function.fixed = True
    
    # start the inner layers deterministically 
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5
    
    return model



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
    
    
m_sgpr =  SGPR(X, Y, RBF(D, variance=Y_std**2, lengthscales=2), Z)
m_sgpr.mean_function = Constant(Y_mean)
m_sgpr.mean_function.fixed = True
m_sgpr.optimize()
print np.average(m_sgpr.predict_density(Xs, Ys))




for L in [1, 2, 3]:
    model = make_dgp(L)
    cb = CB(model, 100)
    if L == 1:
        model.optimize(maxiter=2000, callback=cb.cb)
    else:    
        model.optimize(tf.train.AdamOptimizer(0.01), maxiter=2000, callback=cb.cb)
    
    S = 100
    for _ in range(2):
        m, v = model.predict_y(Xs, S)
        Ys_ = Ys[None, :, :] * np.ones((S, 1, 1))
        logliks = norm.logpdf(Ys_, loc=m, scale=v**0.5)
        a = logsumexp(logliks, axis=0, b=1/float(S))
        #print np.average(a)
        print np.average(logliks)
    
    for _ in range(2):
        ll = model.predict_density(Xs, Ys, S)
        print np.average(ll)
    
    plt.plot(cb.res)
    plt.show()
#
#for n in [1, 10, 100]:
#    print n
#    for _ in range(3):
#        test_log_likelihood = model.predict_density(Xs, Ys, n)
#        a = logsumexp(test_log_likelihood, axis=0, b=1/float(test_log_likelihood.shape[0]))
#        print np.average(a)
##        m, v = m_dgp
#
#
#







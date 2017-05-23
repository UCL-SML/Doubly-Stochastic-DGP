# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:16:38 2017

@author: hrs13
"""

import sys
sys.path.append('../src')
sys.path.append('../../DNSGP/GPflow')

import numpy as np
import tensorflow as tf

from GPflow.likelihoods import MultiClass
from GPflow.kernels import RBF, White, Linear, Matern32, Matern52
from GPflow.svgp import SVGP
from GPflow.gpr import GPR

from GPflow.param import AutoFlow

from scipy.stats import mode
from scipy.cluster.vq import kmeans2

from get_data import get_mnist_data
from dgp import DGP

import time

X, Y, Xs, Ys = get_mnist_data()

M = 100
Z = kmeans2(X, M, minit='points')[0]

class MultiClassSVPG(SVGP):
    @AutoFlow((tf.float64, [None, None]), (tf.int32, [None,]))
    def predict_density(self, Xnew, Ynew):
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)
        
        
m_sgp = MultiClassSVPG(X, Y, RBF(784, lengthscales=2, variance=2), 
             MultiClass(10), Z, 
             num_latent=10, minibatch_size=1000)

def make_dgp(L):
    kernels = [RBF(784, lengthscales=2., variance=2.)]
    for l in range(L-1):
        kernels.append(RBF(30, lengthscales=2., variance=2.))
    model = DGP(X, Y, Z, kernels, MultiClass(10), 
                num_samples=1,
                minibatch_size=1000,
                num_latent_Y=10)

    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5 
    
    return model

m_dgp2 = make_dgp(2)
m_dgp3 = make_dgp(3)

    
    
S = 100

def assess_model_sgp(model, X_batch, Y_batch):
    m, v = model.predict_y(X_batch)
    l = model.predict_density(X_batch, Y_batch)
    a = (np.argmax(m, 1)==Y_batch)
    return l, a
        
def assess_model_dgp(model, X_batch, Y_batch):
    m, v = model.predict_y(X_batch, S)
    l = model.predict_density_multiclass(X_batch, Y_batch, S)
    a = (mode(np.argmax(m, 2), 0)[0].flatten()==Y_batch)
    return l, a
        
def batch_assess(model, assess_model, X, Y):
    n_batches = int(len(X)/100)
    lik, acc = [], []
    for X_batch, Y_batch in zip(np.split(X, n_batches), np.split(Y, n_batches)):
        l, a = assess_model(model, X_batch, Y_batch)
        lik.append(l)
        acc.append(a)
    lik = np.concatenate(lik, 0)
    acc = np.array(np.concatenate(acc, 0), dtype=float)
    return np.average(lik), np.average(acc)
    
    
class CB(object):
    def __init__(self, model, assess_model):
        self.model = model
        self.assess_model = assess_model
        self.i = 0
        self.t = time.time()
        self.train_time = 0
        self.ob = []
        self.train_lik = []
        self.train_acc = []
    def cb(self, x):
        self.i += 1
        if self.i % 1000 == 0:
            # time how long we've be training 
            self.train_time += time.time() - self.t
            self.t = time.time()
            
            # assess the model on the training data
            self.model.set_state(x)
            lik, acc = batch_assess(self.model, self.assess_model, X, Y)
            self.train_lik.append(lik)
            self.train_acc.append(acc)
            
            # calculate the objective, averaged over S samples 
            ob = 0
            for _ in range(S):
                ob += self.model.compute_log_likelihood()/float(S)
            self.ob.append(ob)
            
            st = 'it: {}, ob: {:.1f}, train lik: {:.4f}, train acc {:.4f}'
            print st.format(self.i, ob, lik, acc)
            
            
cb_sgp = CB(m_sgp, assess_model_sgp)
m_sgp.optimize(tf.train.AdamOptimizer(0.01), maxiter=20000, callback=cb_sgp.cb)
print 'sgp total train time {:.4f}'.format(cb_sgp.train_time)
l, a = batch_assess(m_sgp, assess_model_sgp, Xs, Ys)
print 'spg test lik: {:.4f}, test acc {:.4f}'.format(l, a)

cb_dgp2 = CB(m_dgp2, assess_model_dgp)
m_dgp2.optimize(tf.train.AdamOptimizer(0.01), maxiter=20000, callback=cb_dgp2.cb)
print 'dgp2 total train time {:.4f}'.format(cb_dgp2.train_time)
l, a = batch_assess(m_dgp2, assess_model_dgp, Xs, Ys)
print 'dgp2 test lik: {:.4f}, test acc {:.4f}'.format(l, a)

cb_dgp3 = CB(m_dgp3, assess_model_dgp)
m_dgp3.optimize(tf.train.AdamOptimizer(0.01), maxiter=20000, callback=cb_dgp3.cb)
print 'dgp3 total train time {:.4f}'.format(cb_dgp3.train_time)
l, a = batch_assess(m_dgp3, assess_model_dgp, Xs, Ys)
print 'dgp3 test lik: {:.4f}, test acc {:.4f}'.format(l, a)














#
## def test_all(S):
#    
#S = 1000
#t = time.time()
#preds_sgp = []
#for Xs_split in np.split(Xs, 100): # chunks of 100 
#    m, v = m_sgp.predict_y(Xs_split)
#    preds_sgp.append(np.argmax(m, 1))
#preds_sgp = np.concatenate(preds_sgp, 0)    
#acc = np.average(np.array(preds_sgp==Ys, dtype=float))
#print 'time to predict sgp {:.4f}. acc: {}'.format(time.time() - t, acc)
#
#
#t = time.time()
#preds_dgp1= []
#for Xs_split in np.split(Xs, 100): # chunks of 100 
#    m, v = m_dgp1.predict_y(Xs_split, S)
#    preds_dgp1.append(mode(np.argmax(m, 2), 0)[0].flatten())
#preds_dgp1 = np.concatenate(preds_dgp1, 0)    
#acc = np.average(np.array(preds_dgp1==Ys, dtype=float))
#print 'time to predict dgp 1 {:.4f}. acc: {}'.format(time.time() - t, acc)
#
#t = time.time()
#preds_dgp2= []
#for Xs_split in np.split(Xs, 100): # chunks of 100 
#    m, v = m_dgp2.predict_y(Xs_split, S)
#    preds_dgp2.append(mode(np.argmax(m, 2), 0)[0].flatten())
#preds_dgp2 = np.concatenate(preds_dgp2, 0)    
#acc = np.average(np.array(preds_dgp2==Ys, dtype=float))
#print 'time to predict dgp 1 {:.4f}. acc: {}'.format(time.time() - t, acc)
#
#    
#    
#    





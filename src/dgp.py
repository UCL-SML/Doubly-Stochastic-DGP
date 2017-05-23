# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:35:36 2017

@author: hrs13
"""

import tensorflow as tf
import numpy as np

from GPflow.param import Param, ParamList, Parameterized, AutoFlow, DataHolder
from GPflow.minibatch import MinibatchData
from GPflow.conditionals import conditional
from GPflow.model import Model
from GPflow.mean_functions import Linear, Zero
from GPflow.kullback_leiblers import gauss_kl_white
from GPflow._settings import settings

float_type = settings.dtypes.float_type

from utils import normal_sample, shape_as_list, tile_over_samples

class Layer(Parameterized):
    def __init__(self, kern, q_mu, q_sqrt, Z, mean_function):
        Parameterized.__init__(self)
        self.q_mu, self.q_sqrt, self.Z = Param(q_mu), Param(q_sqrt), Param(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.mean_function.fixed = True
        
    def conditional(self, X, full_cov=False):
        mean, var = conditional(X, self.Z, self.kern,
                                self.q_mu, q_sqrt=self.q_sqrt,
                                full_cov=full_cov, whiten=True)
        return mean + self.mean_function(X), var
        
    def multisample_conditional(self, X, full_cov=False):
        S, N, D_in = shape_as_list(X)
        X_flat = tf.reshape(X, [S*N, D_in])
        mean_flat, var_flat = self.conditional(X_flat, full_cov)
        D_out = tf.shape(mean_flat)[1]
        mean = tf.reshape(mean_flat, [S, N, D_out])
        if full_cov is True:
            var = tf.reshape(var_flat, [S, N, N, D_out])
        else:
            var = tf.reshape(var_flat, [S, N, D_out])
        return mean, var
        
    def KL(self):
        return gauss_kl_white(self.q_mu, self.q_sqrt)


def init_layers(X, Z, dims, final_mean_function):
    M = Z.shape[0]
    q_mus, q_sqrts, mean_functions, Zs = [], [], [], []
    X_running, Z_running = X.copy(), Z.copy()
    
    for dim_in, dim_out in zip(dims[:-2], dims[1:-1]):
        if dim_in == dim_out: # identity for same dims
            W = np.eye(dim_in)
        elif dim_in > dim_out: # use PCA mf for stepping down
            _, _, V = np.linalg.svd(X_running, full_matrices=False)
            W = V[:dim_out, :].T
        elif dim_in < dim_out: # identity + pad with zeros for stepping up
            I = np.eye(dim_in)
            zeros = np.zeros((dim_in, dim_out - dim_in))
            W = np.concatenate([I, zeros], 1)

        mean_functions.append(Linear(A=W))
        Zs.append(Z_running.copy())
        q_mus.append(np.zeros((M, dim_out)))
        q_sqrts.append(np.eye(M)[:, :, None] * np.ones((1, 1, dim_out)))
        
        Z_running = Z_running.dot(W)
        X_running = X_running.dot(W)

    # final layer (as before but no mean function)
    mean_functions.append(final_mean_function)
    Zs.append(Z_running.copy())
    q_mus.append(np.zeros((M, dims[-1])))
    q_sqrts.append(np.eye(M)[:, :, None] * np.ones((1, 1, dims[-1])))

    return q_mus, q_sqrts, Zs, mean_functions


class DGP(Model):
    def __init__(self, X, Y, Z, kernels, likelihood, 
                 num_latent_Y=None, 
                 minibatch_size=None, 
                 num_samples=1,
                 mean_function=Zero()):
        Model.__init__(self)

        assert X.shape[0] == Y.shape[0]
        assert Z.shape[1] == X.shape[1]
        assert kernels[0].input_dim == X.shape[1]

        self.num_data, D_X = X.shape
        self.num_samples = num_samples
        self.D_Y = num_latent_Y or Y.shape[1]

        self.dims = [k.input_dim for k in kernels] + [self.D_Y, ]
        q_mus, q_sqrts, Zs, mean_functions = init_layers(X, Z, self.dims, 
                                                         mean_function)
                                                         
        layers = []
        for q_mu, q_sqrt, Z, mean_function, kernel in zip(q_mus, q_sqrts, Zs, 
                                                          mean_functions, 
                                                          kernels):
            layers.append(Layer(kernel, q_mu, q_sqrt, Z, mean_function))
        self.layers = ParamList(layers)

        self.likelihood = likelihood
        
        if minibatch_size is not None:
            self.X = MinibatchData(X, minibatch_size)
            self.Y = MinibatchData(Y, minibatch_size)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

    def propagate(self, X, full_cov=False, S=1):
        Fs = [tile_over_samples(X, S), ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            mean, var = layer.multisample_conditional(Fs[-1], full_cov=full_cov)
            F = normal_sample(mean, var, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars # don't return Fs[0] as this is just X

    def build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov, S)
        return Fmeans[-1], Fvars[-1]
    
    def build_likelihood(self):
        Fmean, Fvar = self.build_predict(self.X, full_cov=False, S=self.num_samples)

        S, N, D = shape_as_list(Fmean)
        Y = tile_over_samples(self.Y, self.num_samples)
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar, Y]]
        var_exp = self.likelihood.variational_expectations(*flat_arrays) #SN
        var_exp = tf.reshape(var_exp, [S, N])
        var_exp = tf.reduce_mean(var_exp, 0) # S,N -> N. Average over samples
        L = tf.reduce_sum(var_exp) # N -> scalar. Sum over data (minibatch)

        KL = 0.
        for layer in self.layers:
            KL += layer.KL()

        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self.build_predict(Xnew, full_cov=False, S=num_samples)
    
    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        return self.build_predict(Xnew, full_cov=True, S=1)
    
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)[0]

    @AutoFlow((float_type, [None, None]))
    def predict_all_layers_full_cov(self, Xnew):
        return self.propagate(Xnew, full_cov=True, S=1)[0]
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self.build_predict(Xnew, full_cov=False, S=num_samples)
        S, N, D = shape_as_list(Fmean)
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar]]
        Y_mean, Y_var = self.likelihood.predict_mean_and_var(*flat_arrays)
        return [tf.reshape(a, [S, N, self.D_Y]) for a in [Y_mean, Y_var]]
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        return self._predict_density(Xnew, Ynew, num_samples)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, [None, ]), (tf.int32, []))
    def predict_density_multiclass(self, Xnew, Ynew, num_samples):
        return self._predict_density(Xnew, Ynew, num_samples)
    
    def _predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self.build_predict(Xnew, full_cov=False, S=num_samples)
        S, N, D = shape_as_list(Fmean)
        Ynew = tile_over_samples(Ynew, num_samples)
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar, Ynew]]
        l_flat = self.likelihood.predict_density(*flat_arrays)
        l = tf.reshape(l_flat, [S, N])
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)









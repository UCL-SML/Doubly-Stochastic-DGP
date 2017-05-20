# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:35:36 2017

@author: hrs13
"""


import GPflow
import tensorflow as tf
import numpy as np

from GPflow.param import Param
from GPflow.param import Parameterized
from GPflow.param import ParamList
from GPflow.conditionals import conditional
from GPflow.model import Model
from GPflow.mean_functions import Linear
from GPflow.kullback_leiblers import gauss_kl_white

from utils import normal_sample

class Layer(Parameterized):
    def __init__(self, kern, Z, dim_out, mean_function):
        M, dim_in = Z.shape
        assert kern.input_dim == dim_in

        self.q_mu = Param(np.zeros(M, dim_out))
        self.q_sqrt = Param(np.eye(M)[:, None] * np.ones((1, 1, dim_out)))

        self.Z = Param(Z)
        self.kern = kern

        self.mean_function = mean_function
        self.mean_function.fixed = True

    def conditional(self, X, full_cov=False):
        mean, var = conditional(X, self.Z, self.kern,
                                self.q_mu, q_sqrt=self.q_sqrt,
                                full_cov=full_cov, whiten=True)
        return mean + self.mean_function(X), var

    def KL(self):
        return gauss_kl_white(self.q_mu, self.q_sqrt)


class DGP(Model):
    def __init__(self, X, Y, Z, kernels, likelihood, num_latent_Y=None):
        assert X.shape[0] == Y.shape[0]
        assert Z.shape[1] == X.shape[1]
        assert kernels[0].input_dim == X.shape[0]

        self.N, self.D_X = X.shape
        self.D_Y = num_latent_Y else Y.shape[1]

        self.dims = [k.input_dim for k in kernels] + [self.D_Y]

        layers = []
        for dim_in, dim_out, kern in zip(self.dims[:-1], self.dims[1:], kernels):
            if dim_in == dim_out:
                W = np.eye(dim_in)
            elif dim_in > dim_out:
                central_Z = Z - np.average(Z, 0)[None, :]
                U, s, V = np.linalg.svd(central_Z, full_matrices=False)
                W = V[:dim_out, :].T

            else:
                raise NotImplementedError

            mf = Linear(W)
            layers.append(Layer(kern, Z.copy(), dim_out, mf))
            Z = Z.dot(W)
        self.layers = ParamList(layers)

        self.likelihood = likelihood


    def propagate(self, Xstar, full_cov=False, S=1):
        N = tf.shape(Xstar)[0]
        Fstars, Fstar_means, Fstar_vars = [], [], []

        if full_cov is True:
            assert S == 1 # can't take full cov across multiple samples with this implementation

        mean, var = self.layer[0].conditional(Xstar, full_cov=full_cov)
        Fstar = normal_sample(S, mean, var, full_cov=full_cov)
        Fstars.append(Fstar)
        Fstar_flat = tf.reshape(Fstar, [S*N, -1])

        for layer in self.layers[1:]:
            mean_flat, var_flat = layer.conditional(Fstar_flat, full_cov=full_cov)
            Fstar_flat = normal_sample(mean_flat, var_flat, full_cov=full_cov)

            D = tf.shape(mean_flat)[2]
            Fstars.append(tf.reshape(Fstar, [S, N, D]))
            Fstar_means.append(tf.reshape(mean, [S, N, D]))
            if full_cov is True:
                Fstar_vars.append(tf.reshape(var, [S, N, D, D]))
            else:
                Fstar_vars.append(tf.reshape(var, [S, N, D]))

        return Fstars, Fstar_means, Fstar_vars

    def build_likelihood(self):
        Fmean, Fvar = self.build_predict(self.X, full_cov=False)

        S, N, D = tf.shape(Fmean)
        Y = tf.ones([S, 1, 1], dtype=tf.float64) * self.Y[None, :, :]

        Fmean_flat = tf.reshape(Fmean, [S*N, D])
        Fvar_flat = tf.reshape(Fvar, [S*N, D])
        Y_flat = tf.reshape(Y, [S*N, -1])
        var_exp_flat = self.likelihood.variational_expectations(Fmean_flat,
                                                                Fvar_flat,
                                                                Y_flat)
        var_exp = tf.reshape(var_exp_flat, [S, N])
        L = tf.reduce_mean(tf.reduce_sum(var_exp, 0))

        KL = 0.
        for layer in self.layers:
            KL += layer.KL()

        scale = tf.cast(self.N, tf.float64)
        scale /= tf.cast(tf.shape(self.X)[0], tf.float64)  # minibatch size
        return L * scale - KL













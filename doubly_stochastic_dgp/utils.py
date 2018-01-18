# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from gpflow import settings
from gpflow.likelihoods import Likelihood
from gpflow import params_as_tensors
import numpy as np

from gpflow.likelihoods import *

def normal_sample(mean, var, full_cov=False):
    z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
    return reparameterize(mean, var, z, full_cov=full_cov)

def reparameterize(mean, var, z, full_cov=False):
    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND


class BroadcastingLikelihood(Parameterized):
    """
    A wrapper for the likelihood to broadcast over the samples dimension. Some likelihoods
    don't need this, but for others (e.g. bernoulli) need reshaping and tiling.
    """
    def __init__(self, likelihood):
        Parameterized.__init__(self)
        self.likelihood = likelihood

        to_broadcast = [Bernoulli, MultiClass]
        if np.any([isinstance(likelihood, L) for L in to_broadcast]):
            self.needs_broadcasting = True
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if self.needs_broadcasting is False:
            return f(vars_SND, vars_ND)

        else:
            S, N, D = [tf.shape(vars_SND[0])[i] for i in range(3)]
            vars_tiled = [tf.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

            flattened_SND = [tf.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [tf.reshape(x, [S*N, D]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [tf.reshape(x, [S, N, D]) for x in flattened_result]
            else:
                return tf.reshape(flattened_result, [S, N, D])

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                                vars_SND[1],
                                                                                vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])

    @params_as_tensors
    def logp(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    @params_as_tensors
    def conditional_mean(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
         return self._broadcast(f,[F], [])

    @params_as_tensors
    def conditional_variance(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
         return self._broadcast(f,[F], [])

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
         f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0],
                                                                             vars_SND[1])
         return self._broadcast(f,[Fmu, Fvar], [])

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_density(vars_SND[0],
                                                                       vars_SND[1],
                                                                       vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])



class PositiveTransform(object):
    eps = 1e-6
    def forward(self, x):
        NotImplementedError

    def backward(self, y):
        NotImplementedError

    def forward_np(self, x):
        NotImplementedError

    def backward_np(self, y):
        NotImplementedError


class PositiveSoftplus(PositiveTransform):
    def forward(self, x):
        result = tf.log(1. + tf.exp(x)) + self.eps
        return tf.where(x > 35., x + self.eps, result)

    def backward(self, y):
        result = tf.log(tf.exp(y - self.eps) - 1.)
        return tf.where(y > 35., y - self.eps, result)

    def forward_np(self, x):
        result = np.log(1. + np.exp(x)) + self.eps
        return np.where(x > 35., x + self.eps, result)

    def backward_np(self, y):
        result = np.log(np.exp(y - self.eps) - 1.)
        return np.where(y > 35., y - self.eps, result)


class PositiveExp(PositiveTransform):
    def forward(self, x):
        return tf.exp(x) + self.eps

    def backward(self, y):
        return tf.log(y - self.eps)

    def forward_np(self, x):
        return np.exp(x) + self.eps

    def backward_np(self, y):
        return np.log(y - self.eps)

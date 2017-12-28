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

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors
from gpflow.models.model import Model
from gpflow.mean_functions import Zero
from gpflow.likelihoods import Gaussian
from gpflow import settings

float_type = settings.float_type

from doubly_stochastic_dgp.utils import normal_sample
from doubly_stochastic_dgp.layer_initializations import init_layers_linear_mean_functions

class DGP(Model):
    def __init__(self, X, Y, Z, kernels, likelihood, 
                 num_latent_Y=None, 
                 minibatch_size=None, 
                 num_samples=1,
                 mean_function=Zero(),
                 init_layers=init_layers_linear_mean_functions,
                 analytic_final_expectations=False):
        Model.__init__(self)
        self.num_data = X.shape[0]
        self.num_samples = num_samples
        self.D_Y = num_latent_Y or Y.shape[1]
        self.analytic_final_expectations = analytic_final_expectations
        self.layers = init_layers(X, Y, Z, kernels, self.D_Y)
        self.layers[-1].mean_function = mean_function

        self.likelihood = likelihood
        
        if minibatch_size is not None:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
        Fs = [sX, ]
        Fmeans, Fvars = [], []

        for layer in self.layers:
            if layer.forward_propagate_inputs:
                X_inputs = tf.concat([sX, Fs[-1]], 2)
            else:
                X_inputs = Fs[-1]

            mean, var = layer.multisample_conditional(X_inputs, full_cov=full_cov)
            F = normal_sample(mean, var, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars  # don't return Fs[0] as this is just X

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov, S)
        return Fmeans[-1], Fvars[-1]

    @params_as_tensors
    def _build_likelihood(self):
        _, Fmeans, Fvars = self.propagate(self.X, full_cov=False, S=self.num_samples)

        Y = tf.tile(tf.expand_dims(self.Y, 0), [self.num_samples, 1, 1])

        if isinstance(self.likelihood, Gaussian):
            if len(self.layers) == 1 or (not self.analytic_final_expectations):
                Fmean, Fvar = Fmeans[-1], Fvars[-1]

            else:  # compute expectations analytically through final layer
                m, v = Fmeans[-2], Fvars[-2]  # pnultimate layer means and vars
                final_layer = self.layers[-1]
                if final_layer.forward_propagate_inputs:
                    sX = tf.tile(tf.expand_dims(self.X, 0), [self.num_samples, 1, 1])
                    m = tf.concat([sX, m], -1)
                    zeros = 1e-6 * tf.ones(tf.shape(sX), dtype=settings.float_type)
                    v = tf.concat([zeros, v], -1)

                Fmean, Fvar = final_layer.multisample_uncertain_conditional(m, v)

            var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)

        else:
            f = lambda a: self.likelihood.variational_expectations(a[0], a[1], a[2])
            var_exp = tf.stack(tf.map_fn(f, (Fmeans[-1], Fvars[-1], Y), dtype=float_type))  # S,N

        L = tf.reduce_sum(tf.reduce_mean(var_exp, 0))  # S,N -> N -> scalar

        KL = 0.
        for layer in self.layers:
            KL += layer.KL()

        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)
    
    @autoflow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        return self._build_predict(Xnew, full_cov=True, S=1)
    
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)
    
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        S, N, D = tf.shape(Fmean)[0], tf.shape(Fmean)[1], tf.shape(Fmean)[2]
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar]]
        Y_mean, Y_var = self.likelihood.predict_mean_and_var(*flat_arrays)
        return [tf.reshape(a, [S, N, self.D_Y]) for a in [Y_mean, Y_var]]
    
    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        S, N, D = tf.shape(Fmean)[0], tf.shape(Fmean)[1], tf.shape(Fmean)[2]
        Ynew = tf.tile(tf.expand_dims(Ynew, 0), [S, 1, 1])
        flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar, Ynew]]
        l_flat = self.likelihood.predict_density(*flat_arrays)
        l = tf.reshape(l_flat, [S, N, -1])
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

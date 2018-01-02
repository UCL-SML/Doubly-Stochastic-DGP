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
from doubly_stochastic_dgp.layer import Layer
from doubly_stochastic_dgp.utils import PositiveSoftplus

class DGP(Model):
    """
    This is the Doubly-Stochastic Deep GP. The key reference is

    ::

      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels, likelihood, 
                 num_latent_Y=None, 
                 minibatch_size=None, 
                 num_samples=1,
                 mean_function=Zero(),
                 init_layers=init_layers_linear_mean_functions):
        Model.__init__(self)

        # shapes
        self.num_data = X.shape[0]
        self.num_samples = num_samples
        self.D_Y = num_latent_Y or Y.shape[1]

        # DGP layers
        self.layers = init_layers(X, Y, Z, kernels, self.D_Y)
        self.layers[-1].mean_function = mean_function

        self.likelihood = likelihood

        # data
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
        Fmean, Fvar = self._build_predict(self.X, full_cov=False, S=self.num_samples)

        if (isinstance(self.likelihood, Gaussian) or
            isinstance(self.likelihood, HeteroscedasticGaussianlikelihood)):
            # the Gaussian likelihood broadcasts correctly so no need to tile or use map_fn
            var_exp = self.likelihood.variational_expectations(Fmean, Fvar, self.Y[None, :, :])
        else:
            Y = tf.tile(self.Y[None, :, :], [self.num_samples, 1, 1])
            f = lambda a: self.likelihood.variational_expectations(a[0], a[1], a[2])
            var_exp = tf.stack(tf.map_fn(f, (Fmean, Fvar, Y), dtype=float_type))  # S,N

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


from gpflow.params import Parameter
from gpflow import transforms
from gpflow.likelihoods import Likelihood
from gpflow import densities
import numpy as np


class HeteroscedasticGaussianlikelihood(Likelihood):
    def __init__(self, Dy, v_0=0.1, positive_transform=PositiveSoftplus()):
        Likelihood.__init__(self)
        self.Dy = Dy
        self.positive_transform=positive_transform
        self.v_0 = Parameter(v_0, transform=transforms.positive)


    def _make_l_positive(self, l):
        return self.positive_transform.forward(l + self.positive_transform.backward(self.v_0))

    def _slice_variance(self, F_l):
        dims = F_l.get_shape().ndims
        start_F = [0, ] * dims
        end_F = [-1, ] * (dims - 1) + [self.Dy, ]

        start_l = [0, ] * (dims - 1) + [self.Dy]
        end_l = [-1, ] * (dims - 1) + [-1, ]

        # l = F_l[:, :, self.Dy:]
        # F = F_l[:, :, :self.Dy]

        F = tf.slice(F_l, start_F, end_F)
        l = tf.slice(F_l, start_l, end_l)


        return F, self._make_l_positive(l)

    def _slice_and_sample(self, F_mu_l, F_var_l, full_cov=False):
        F_mu, l_mu = self._slice_variance(F_mu_l)
        F_var, l_var = self._slice_variance(F_var_l)
        l = normal_sample(l_mu, l_var, full_cov=full_cov)
        return F_mu, F_var, self._make_l_positive(l)

    @params_as_tensors
    def logp(self, F_l, Y):
        F, variance = self._slice_variance(F_l)
        return densities.gaussian(F, Y, variance)

    @params_as_tensors
    def conditional_mean(self, F_l):
        F, variance = self._slice_variance(F_l)
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F_l):
        F, variance = self._slice_variance(F_l)
        return tf.fill(tf.shape(F), tf.squeeze(variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu_l, Fvar_l):
        Fmu, Fvar, variance = self._slice_and_sample(Fmu_l, Fvar_l)
        return tf.identity(Fmu), Fvar + variance

    @params_as_tensors
    def predict_density(self, Fmu_l, Fvar_l, Y):
        Fmu, Fvar, variance = self._slice_and_sample(Fmu_l, Fvar_l)
        return densities.gaussian(Fmu, Y, Fvar + variance)

    @params_as_tensors
    def variational_expectations(self, Fmu_l, Fvar_l, Y):
        Fmu, Fvar, variance = self._slice_and_sample(Fmu_l, Fvar_l)
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / variance


class HeteroscedasticDGP(DGP):
    # def __init__(self, X, Y, Z, kernels, likelihood, likelihood_kernel, **kwargs):
    def __init__(self, X, Y, Z, kernels, likelihood, likelihood_kernel, **kwargs):

        assert isinstance(likelihood, HeteroscedasticGaussianlikelihood)
        DGP.__init__(self, X, Y, Z, kernels, likelihood, **kwargs)

        M, D_Y = Z.shape[0], Y.shape[1]
        q_mu = np.zeros((M, D_Y))
        q_sqrt = np.tile(np.eye(M)[:, :, None], [1, 1, D_Y])
        self.likelihood_variance_layer = Layer(likelihood_kernel, q_mu, q_sqrt, Z, Zero())

    def propagate(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = DGP.propagate(self, X, full_cov=full_cov, S=S)

        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
        l_mean, l_var = self.likelihood_variance_layer.multisample_conditional(sX, full_cov)

        Fmeans[-1] = tf.concat([Fmeans[-1], l_mean], -1)

        Fs[-1] = tf.concat([Fs[-1], normal_sample(l_mean, l_var, full_cov=full_cov)], -1)
        Fvars[-1] = tf.concat([Fvars[-1], l_var], -1)

        # if full_cov is True:
        #     l_var_SDNN = tf.transpose(l_var, [0, 2, 3, 1])
        #     l_var_SDN = tf.matrix_diag_part(l_var_SDNN)
        #     l_var_SND = tf.transpose(l_var_SDN, [0, 2, 1])
        #
        #     Fs[-1] = tf.concat([Fs[-1], normal_sample(l_mean, l_var_SND, full_cov=False)], -1)
        #
        #     l_var_diag_SDNN = tf.matrix_diag(l_var_SDN)
        #     l_var_diag_SNND = tf.transpose(l_var_diag_SDNN, [0, 2, 3, 1])
        #     Fvars[-1] = tf.concat([Fvars[-1], l_var_diag_SNND], -1)
        #
        # else:
        #     Fs[-1] = tf.concat([Fs[-1], normal_sample(l_mean, l_var, full_cov=False)], -1)
        #     Fvars[-1] = tf.concat([Fvars[-1], l_var], -1)



        # if full_cov is False:
        #     Fvars[-1] = tf.concat([Fvars[-1], l_var], -1)
        # else:
        #     l_var_SDN = tf.transpose(l_var, [0, 2, 1])
        #     l_var_SDNN = tf.matrix_diag(l_var_SDN)
        #     l_var_SNND = tf.transpose(l_var_SDNN, [0, 2, 3, 1])
        #     Fvars[-1] = tf.concat([Fvars[-1], l_var_SNND], -1)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_likelihood(self):
        return DGP._build_likelihood(self) - self.likelihood_variance_layer.KL()

    @params_as_tensors
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_likelihood_variance(self, Xnew, num_samples):
        Fs, Fmeans, Fvars = self.propagate(Xnew, S=num_samples)
        return self.likelihood._slice_variance(Fs[-1])[1]

    @params_as_tensors
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_likelihood_variance_full_cov(self, Xnew, num_samples):
        Fs, Fmeans, Fvars = self.propagate(Xnew, full_cov=True, S=num_samples)
        return self.likelihood._slice_variance(Fs[-1])[1]

# class EDGP(DGP):
#     """
#     An enhanced DGP model, with analytic expectations for the final layer (only with
#     Gaussian likelihood) and additional GPs for noise and amplitude variance
#
#     """
#     def __init__(self, X, Y, Z, kernels, likelihood,
#                  analytic_final_expectations=False,
#                  amplitude_scaling_layer=IdentityScalingLayer(1.),
#                  additive_variance_layer=IdentityScalingLayer(0.),
#                  **kwargs):
#         DGP.__init__(self,  X, Y, Z, kernels, likelihood, **kwargs)
#
#
#         assert isinstance(likelihood, Gaussian)
#         self.likelihood_min = 1e-6
#
#         del likelihood.variance
#         likelihood.variance = None
#
#         self.analytic_final_expectations = analytic_final_expectations
#
#         # additional GP layers for heteroscedastic amplitude and noise variance
#         self.amplitude_scaling_layer = amplitude_scaling_layer
#         self.additive_variance_layer = additive_variance_layer
#
#
#     @params_as_tensors
#     def propagate(self, X, full_cov=False, S=1):
#         Fs, Fmeans, Fvars = DGP.propagate(self, X, full_cov=full_cov, S=S)
#         sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
#         a = self.amplitude_scaling_layer.scale(sX, full_cov=full_cov)
#         # v = self.additive_variance_layer.scale(sX, full_cov=full_cov)
#         a += self.likelihood_min
#
#         Fmeans[-1] = Fmeans[-1] * tf.sqrt(a) # scale the mean by the standard deviation
#
#         Fs[-1] = Fs[-1] * tf.sqrt(a)# + tf.sqrt(v) * tf.random_normal(tf.shape(v), dtype=settings.float_type) # scale the samples by the standard deviation
#         # and add a sample from the noise
#
#         if full_cov is False:
#             Fvars[-1] = Fvars[-1] * a #+ v  # scale by the variance, and add on noise
#
#         else:
#             Fvar_final = Fvars[-1]
#
#             if a.get_shape().ndims != 0:
#                 Fvar_final *= a[:, :, None, :]
#             else:
#                 Fvar_final *= a
#
#             # if v.get_shape().ndims != 0:
#             #     v = tf.transpose(v, [0, 2, 1])  # S,D,N
#             #     v = tf.matrix_diag(v)  # S,D,N,N
#             #     v = tf.transpose(v, [0, 2, 3, 1])  # S,N,N,D
#             #     Fvar_final += v
#             # else:
#             #     Fvar_final += v
#
#             Fvars[-1] = Fvar_final
#
#         return Fs, Fmeans, Fvars
#
#     @params_as_tensors
#     def _build_likelihood(self):
#         Fs, Fmeans, Fvars = self.propagate(self.X, full_cov=False, S=self.num_samples)
#
#         Y = tf.tile(tf.expand_dims(self.Y, 0), [self.num_samples, 1, 1])
#
#         if isinstance(self.likelihood, Gaussian):
#             if len(self.layers) == 1 or (not self.analytic_final_expectations):
#                 Fmean, Fvar = Fmeans[-1], Fvars[-1]
#
#             else:  # compute expectations analytically through final layer
#                 m, v = Fmeans[-2], Fvars[-2]  # pnultimate layer means and vars
#                 final_layer = self.layers[-1]
#                 if final_layer.forward_propagate_inputs:
#                     sX = tf.tile(tf.expand_dims(self.X, 0), [self.num_samples, 1, 1])
#                     m = tf.concat([sX, m], -1)
#                     zeros = 1e-6 * tf.ones(tf.shape(sX), dtype=settings.float_type)
#                     v = tf.concat([zeros, v], -1)
#
#                 Fmean, Fvar = final_layer.multisample_uncertain_conditional(m, v)
#
#             sX = tf.tile(tf.expand_dims(self.X, 0), [self.num_samples, 1, 1])
#             v_sample = self.additive_variance_layer.scale(sX, full_cov=False)
#             self.likelihood.variance = v_sample + self.likelihood_min
#
#             var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)
#
#         else:
#             f = lambda a: self.likelihood.variational_expectations(a[0], a[1], a[2])
#             var_exp = tf.stack(tf.map_fn(f, (Fmeans[-1], Fvars[-1], Y), dtype=float_type))  # S,N
#
#         L = tf.reduce_sum(tf.reduce_mean(var_exp, 0))  # S,N -> N -> scalar
#
#         KL = 0.
#         for layer in self.layers:
#             KL += layer.KL()
#
#         for layer in [self.additive_variance_layer, self.amplitude_scaling_layer]:
#             KL += layer.KL()
#
#         scale = tf.cast(self.num_data, float_type)
#         scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
#         return L * scale - KL
#
#     @params_as_tensors
#     @autoflow((float_type, [None, None]), (tf.int32, []))
#     def predict_v(self, Xnew, num_samples):
#         sXnew = tf.tile(tf.expand_dims(Xnew, 0), [num_samples, 1, 1])
#         v = self.additive_variance_layer.scale(sXnew, full_cov=True)
#         return v
#
#     @params_as_tensors
#     @autoflow((float_type, [None, None]), (tf.int32, []))
#     def predict_a(self, Xnew, num_samples):
#         sXnew = tf.tile(tf.expand_dims(Xnew, 0), [num_samples, 1, 1])
#         a = self.amplitude_scaling_layer.scale(sXnew, full_cov=True)
#         return a
#
#     @params_as_tensors
#     @autoflow((float_type, [None, None]), (tf.int32, []))
#     def predict_y(self, Xnew, num_samples):
#         Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
#         S, N, D = tf.shape(Fmean)[0], tf.shape(Fmean)[1], tf.shape(Fmean)[2]
#         flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar]]
#
#         sXnew = tf.tile(tf.expand_dims(Xnew, 0), [num_samples, 1, 1])
#         v = self.additive_variance_layer.scale(sXnew, full_cov=False)
#         self.likelihood.variance = v + self.likelihood_min
#
#         Y_mean, Y_var = self.likelihood.predict_mean_and_var(*flat_arrays)
#         return [tf.reshape(a, [S, N, self.D_Y]) for a in [Y_mean, Y_var]]
#
#     @params_as_tensors
#     @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
#     def predict_density(self, Xnew, Ynew, num_samples):
#         Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
#         S, N, D = tf.shape(Fmean)[0], tf.shape(Fmean)[1], tf.shape(Fmean)[2]
#         Ynew = tf.tile(tf.expand_dims(Ynew, 0), [S, 1, 1])
#         flat_arrays = [tf.reshape(a, [S*N, -1]) for a in [Fmean, Fvar, Ynew]]
#
#
#         sXnew = tf.tile(tf.expand_dims(Xnew, 0), [num_samples, 1, 1])
#         v = self.additive_variance_layer.scale(sXnew, full_cov=False)
#         self.likelihood.variance = v + self.likelihood_min
#
#         l_flat = self.likelihood.predict_density(*flat_arrays)
#         l = tf.reshape(l_flat, [S, N, -1])
#         log_num_samples = tf.log(tf.cast(num_samples, float_type))
#         return tf.reduce_logsumexp(l - log_num_samples, axis=0)

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
import numpy as np

from gpflow.params import DataHolder, Minibatch, Parameterized
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow.likelihoods import Gaussian
from gpflow import settings
float_type = settings.float_type

from doubly_stochastic_dgp.utils import reparameterize
from doubly_stochastic_dgp.dgp import DGP_Base

from doubly_stochastic_dgp.utils import BroadcastingLikelihood
from doubly_stochastic_dgp.layer_initializations import init_layers_linear
from doubly_stochastic_dgp.layers import GPR_Layer, SGPMC_Layer, GPMC_Layer, SVGP_Layer
from gpflow.models import GPModel


# class DGP_Collapsed(DGP_Quad):
#     def __init__(self, X, Y, likelihood, layers,
#                  num_samples=1,
#                  H=100):
#         assert isinstance(likelihood, Gaussian)
#         DGP_Quad.__init__(self, X, Y, likelihood, layers[:-1],
#                           H=H)
#
#         # Model.__init__(self)
#         self.num_samples = num_samples
#
#         assert isinstance(likelihood, Gaussian)
#
#         # extract last layer
#         kern = layers[-1].kern
#         Z = layers[-1].feature.Z.read_value()
#         num_outputs = Y.shape[1]
#         mean_function = layers[-1].mean_function
#
#         self.last_layer = Collapsed_Layer(kern, Z, num_outputs, mean_function)
#
#     @params_as_tensors
#     def propagate(self, X, full_cov=False, S=1, zs=None):
#         SX = tf.tile(X[None, :, :], [S, 1, 1])
#         SX_data = tf.tile(self.X[None, :, :], [S, 1, 1])
#
#         if not self.layers:
#             Fs, Fmeans, Fvars = [], [], []
#             F = SX
#             F_data = SX_data
#
#         else:
#             Fs, Fmeans, Fvars = DGP_Quad.propagate(self, X, full_cov=full_cov, S=S, zs=zs)
#             Fs_data, _, _ = DGP_Quad.propagate(self, self.X, full_cov=full_cov, zs=zs, S=S)
#
#             F = Fs[-1]
#             F_data = Fs_data[-1]
#
#         def single_sample(args):
#             xs, x_mean = args
#             return self.last_layer.build_predict(xs, x_mean, None, self.Y,
#                                                  self.likelihood.likelihood.variance,
#                                                  full_cov=full_cov)
#
#         m, v = tf.map_fn(single_sample,
#                          [F, F_data],
#                          dtype=(tf.float64, tf.float64))
#
#         z = tf.random_normal(tf.shape(m), dtype=settings.float_type)
#         Fs.append(reparameterize(m, v, full_cov=full_cov, z=z))
#         Fmeans.append(m)
#         Fvars.append(v)
#
#         return Fs, Fmeans, Fvars
#
#     @params_as_tensors
#     def _build_likelihood(self):
#         if len(self.layers) == 0:
#             return self.last_layer.build_likelihood(self.X, None, self.Y, self.likelihood.likelihood.variance)
#
#         else:
#
#             XFs, XFmeans, XFvars = DGP_Quad.propagate(self, self.X, full_cov=False, zs=self.gh_x,
#                                                       S=self.H ** self.D_quad)
#
#             def single_sample(args):
#                 x_mean, x_var = args
#                 return self.last_layer.build_likelihood(x_mean, x_var, self.Y, self.likelihood.likelihood.variance)
#
#             L = tf.reduce_sum(tf.map_fn(single_sample, (XFmeans[-1], XFvars[-1]),
#                                         dtype=settings.float_type))
#
#             if len(self.layers):
#                 KL = tf.reduce_sum([layer.KL() for layer in self.layers])
#             else:
#                 KL = 0.
#
#             return L - KL



class DGP_Collapsed(DGP_Base):
    @params_as_tensors
    def inner_layers_propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        if len(self.layers)==1:
            return [sX], [sX], [tf.zeros_like(sX)]

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers[:-1], zs[:-1]):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        _, ms, vs = self.inner_layers_propagate(self.X, full_cov=full_cov, zs=zs)
        self.layers[-1].set_data(ms[-1][0], vs[-1][0], self.Y, self.likelihood.likelihood.variance)
        return DGP_Base.propagate(self, X, full_cov=full_cov, S=S, zs=zs)

    @params_as_tensors
    def _build_likelihood(self):
        _, ms, vs = self.inner_layers_propagate(self.X, full_cov=False)
        self.layers[-1].set_data(ms[-1][0], vs[-1][0], self.Y, self.likelihood.likelihood.variance)
        KL = tf.cast(tf.reduce_sum([layer.KL() for layer in self.layers[:-1]]), dtype=settings.float_type)
        return self.layers[-1].build_likelihood() - KL


class DGP_Heinonen(DGP_Collapsed):
    """
    A dense 2 layer DGP, with HMC for inference over the inner layer
    
    This is only applicable for 2 layer case with a Gaussian likelihood

    This is based on the following paper: 

    @inproceedings{heinonen2016non,
      title={Non-stationary gaussian process regression with hamiltonian monte carlo},
      author={Heinonen, Markus and Mannerstr{\"o}m, Henrik and Rousu, Juho and Kaski, Samuel and L{\"a}hdesm{\"a}ki, Harri},
      booktitle={Artificial Intelligence and Statistics},
      year={2016}
    }

    """

    def __init__(self, X, Y, likelihood, layers, **kwargs):
        assert len(layers) == 2
        assert isinstance(likelihood, Gaussian)
        assert isinstance(layers[0], GPMC_Layer)
        assert isinstance(layers[1], GPR_Layer)
        # layer0 = GPMC_Layer(kernels[0], X, inner_layer_dim, mean_functions[0])
        # layer1 = GPR_Layer(kernels[1], mean_functions[1], Y.shape[1])
        DGP_Collapsed.__init__(self, X, Y, likelihood, layers, **kwargs)

    @params_as_tensors
    def inner_layers_propagate(self, X, full_cov=False, S=1, zs=None):
        f = self.layers[0].build_latents()[None, :, :]
        return [f], [f], [tf.zeros_like(f)]

    # @params_as_tensors
    # def propagate(self, X, full_cov=False, S=1, zs=None):
    #     f_inner = self.layers[0].build_latents()
    #     self.layers[1].set_data(f_inner, self.Y, self.likelihood.likelihood.variance)
    #     return DGP_Base.propagate(self, X, full_cov=full_cov, S=S, zs=zs)
    #
    # @params_as_tensors
    # def _build_likelihood(self):
    #     f_inner = self.layers[0].build_latents()
    #     return self.layers[1].build_likelihood(f_inner, self.Y, self.likelihood.likelihood.variance)





# class DGP_Damianou(Parameterized):
#     """
#     The inference from
#
#     @inproceedings{damianou2013deep,
#       title={Deep gaussian processes},
#       author={Damianou, Andreas and Lawrence, Neil},
#       booktitle={Artificial Intelligence and Statistics},
#       pages={207--215},
#       year={2013}
#     }
#
#     """
#
#
#

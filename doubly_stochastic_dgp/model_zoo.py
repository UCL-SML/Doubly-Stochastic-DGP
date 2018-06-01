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

from gpflow import params_as_tensors
from gpflow.likelihoods import Gaussian
from gpflow import settings

from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import GPR_Layer, GPMC_Layer


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
        Fs, ms, vs = self.inner_layers_propagate(self.X, full_cov=full_cov, zs=zs)
        # self.layers[-1].set_data(ms[-1][0], vs[-1][0], self.Y, self.likelihood.likelihood.variance)
        self.layers[-1].set_data(Fs[-1][0], None, self.Y, self.likelihood.likelihood.variance)
        return DGP_Base.propagate(self, X, full_cov=full_cov, S=S, zs=zs)

    @params_as_tensors
    def _build_likelihood(self):
        Fs, ms, vs = self.inner_layers_propagate(self.X, full_cov=False)
        # self.layers[-1].set_data(ms[-1][0], vs[-1][0], self.Y, self.likelihood.likelihood.variance)
        self.layers[-1].set_data(Fs[-1][0], None, self.Y, self.likelihood.likelihood.variance)
        KL = tf.cast(tf.reduce_sum([layer.KL() for layer in self.layers[:-1]]), dtype=settings.float_type)
        return self.layers[-1].build_likelihood() - KL


class DGP_Heinonen(DGP_Collapsed):
    """
    A dense 2 layer DGP, with HMC for inference over the inner layer
    
    This is only applicable for 2 layer case with a Gaussian likelihood and no minibatches

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
        if 'minibatch_size' in kwargs:
            assert kwargs['minibatch_size'] is None
        DGP_Collapsed.__init__(self, X, Y, likelihood, layers, **kwargs)

    @params_as_tensors
    def inner_layers_propagate(self, X, full_cov=False, S=1, zs=None):
        f = self.layers[0].build_latents()[None, :, :]
        return [f], [f], [tf.zeros_like(f)]




# TODO
# class DGP_Damianou(Parameterized):
#     """
#     The inference from
#
#     @inproceedings{damianou2013deep,
#       title={Deep gaussian processes},
#       author={Damianou, Andreas and Lawrence, Neil},
#       booktitle={Artificial Intelligence and Statistics},
#       year={2013}
#     }
#
#     """
#

class DGP_Joint_Collapsed(DGP_Base):
    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])
        Ns = tf.shape(X)[0]

        sDataX = tf.tile(tf.expand_dims(self.X, 0), [S, 1, 1])

        joint_Fs, joint_Fmeans, joint_Fvars = [], [], []

        F = tf.concat([sX, sDataX], 1)  # S, Ns + N, D

        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers[:-1], zs[:-1]):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            joint_Fs.append(F)
            joint_Fmeans.append(Fmean)
            joint_Fvars.append(Fvar)


        Fs = [F[:, :Ns, :] for F in joint_Fs]
        Fmeans = [F[:, :Ns, :] for F in joint_Fmeans]
        if full_cov:
            Fvars = [F[:, :Ns, :Ns, :] for F in joint_Fvars]
        else:
            Fvars = [F[:, :Ns, :] for F in joint_Fvars]

#+ 0*joint_Fs[-1][0, Ns:, :]
        # XX = tf.concat([self.X, tf.zeros((20, 1), dtype=settings.float_type)], 1)
        # self.layers[-1].set_data(joint_Fs[-1][0, Ns:, :], None, self.Y, self.likelihood.likelihood.variance)

        # Fs[-1] = tf.reshape(Fs[-1], [S, Ns, 2])
        # XXs = tf.concat([sX, 0*Fs[-1][:, :, 1:]], 2)

        F, Fmean, Fvar = self.layers[-1].ssample_from_conditional(Fs[-1],
                                                                 joint_Fs[-1][:, Ns:, :],
                                                                 self.Y,
                                                                 self.likelihood.likelihood.variance,
                                                                 z=zs[-1], full_cov=full_cov)
        # F, Fmean, Fvar = self.layers[-1].sample_from_conditional(XXs, z=zs[-1], full_cov=full_cov)

        Fs.append(F)
        Fmeans.append(Fmean)
        Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_likelihood(self):
        Fs, ms, vs = self.propagate(self.X, S=self.num_samples, full_cov=False)
        KL = tf.cast(tf.reduce_sum([layer.KL() for layer in self.layers[:-1]]), dtype=settings.float_type)
        L = self.layers[-1].sbuild_likelihood(Fs[-2], self.Y, self.likelihood.likelihood.variance)
        return L - KL
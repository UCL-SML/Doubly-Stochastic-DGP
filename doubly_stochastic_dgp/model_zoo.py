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

from gpflow import params_as_tensors
from gpflow.likelihoods import Gaussian
from gpflow import settings
from gpflow.mean_functions import Zero
from gpflow.params import Parameter
from gpflow import transforms

from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import GPR_Layer, GPMC_Layer, SGPR_Layer


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
        assert isinstance(layers[1], GPR_Layer) or isinstance(layers[1], SGPR_Layer)
        if 'minibatch_size' in kwargs:
            assert kwargs['minibatch_size'] is None
        DGP_Collapsed.__init__(self, X, Y, likelihood, layers, **kwargs)

    @params_as_tensors
    def inner_layers_propagate(self, X, full_cov=False, S=1, zs=None):
        f = self.layers[0].build_latents()[None, :, :]
        return [f], [f], [None]





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


class GaussianWithVarY(Gaussian):
    """
    As Gaussian likelihood, but with variance for Y as well as F
    """
    def variational_expectations(self, Fmu, Fvar, Ymean, Yvar):
        return Gaussian.variational_expectations(self, Fmu, Fvar, Ymean) - 0.5 * Yvar / self.variance

from gpflow import mean_functions

def run_through_mean_functions(X, layers):
    ret = []

    def prop(mf, X):
        if isinstance(mf, mean_functions.Identity):
            pass
        elif isinstance(mf, mean_functions.Zero):
            X = 0*X
        elif isinstance(mf, mean_functions.Linear):
            X = X @ mf.A.read_value()

        ret.append(X.copy())
        return X

    for layer in layers[:-1]:
        X = prop(layer.mean_function, X)

    return ret

class DGP_Damianou(DGP_Base):
    """
    The inference from

    @inproceedings{damianou2013deep,
      title={Deep gaussian processes},
      author={Damianou, Andreas and Lawrence, Neil},
      booktitle={Artificial Intelligence and Statistics},
      year={2013}
    }
    
    but for minibatches. Using natural gradients (step size 1) for each layer recovers the original approach.
    
    For prediction it is not immediately clear how to treat q_X*, the variational distribution of the inputs. We
    simply use the approach of adding the relevant amount of noise and, i.e. we set q_X* to the 
    marginals of the forward propagated outputs and sample from p(g|f)

    NB this doesn't work with input propagation

    """
    def __init__(self, X, Y, likelihood, layers,
                 minibatch_size=None,
                 num_samples=1, num_data=None,
                 **kwargs):
        assert num_samples == 1
        assert minibatch_size is None

        N = X.shape[0]

        # the qX distribution, for all but the first layer

        mean_inits = run_through_mean_functions(X.copy(), layers)

        for layer, mean_init, prev_layer in zip(layers[1:], mean_inits, layers[:-1]):
            if prev_layer.input_prop_dim:
                D = layer.kern.input_dim - prev_layer.input_prop_dim
            else:
                D = layer.kern.input_dim
            print(D)
            layer.q_X_mu = Parameter(mean_init[:, -D:])
            layer.q_X_sqrt = Parameter(1e-5 * np.ones((N, D)), transform=transforms.positive)
            print(layer.q_X_mu.shape)
            print(layer.q_X_sqrt.shape)

        # the between layer Gaussian noise, for all but the final layer
        for layer in layers[:-1]:
            layer.between_layer_likelihood = GaussianWithVarY()
            layer.between_layer_likelihood.variance = 1e-5

        DGP_Base.__init__(self, X, Y, likelihood, layers,
                          num_data=num_data, **kwargs)


    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []
        F = sX
        zs = zs or [None, ] * len(self.layers)
        for l, (layer, z) in enumerate(zip(self.layers, zs)):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            if l != len(self.layers) - 1:
                zz =  tf.random_normal(tf.shape(F), dtype=settings.float_type)
                F += layer.between_layer_likelihood.variance**0.5 * zz

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_likelihood(self):
        L = 0.

        for l, layer_in in enumerate(self.layers):
            if l == 0:
                m_out, v_out = layer_in.conditional_ND(self.X, full_cov=False)
            else:
                m_in = layer_in.q_X_mu
                v_in = layer_in.q_X_sqrt**2
                m_out, v_out = layer_in.uncertain_conditional_ND(m_in, tf.matrix_diag(v_in), full_cov=False)

            if l != len(self.layers) - 1:
                layer_out = self.layers[l + 1]
                m_next = layer_out.q_X_mu
                v_next = layer_out.q_X_sqrt ** 2
                E_loglik = layer_in.between_layer_likelihood.variational_expectations
                L += tf.reduce_sum(E_loglik(m_out, v_out, m_next, v_next))

            else:
                E_loglik = self.likelihood.likelihood.variational_expectations
                L += tf.reduce_sum(E_loglik(m_out, v_out, self.Y))

        KL = tf.reduce_sum([layer.KL() for layer in self.layers])

        # entropy term for q(X)
        ent = tf.reduce_sum([0.5*(np.log(2*np.pi) + 1) + tf.log(layer.q_X_sqrt) for layer in self.layers[1:]])

        return L - KL + ent


class DGP_Damianou_Sampled(DGP_Damianou):
    """
    As DGP_Damianou but without the expensive kernel expectations, using sampling instead.

    This supports input propagation
    """
    @params_as_tensors
    def _build_likelihood(self):
        L = 0.

        for l, layer_in in enumerate(self.layers):
            if l == 0:  # start with ordinary conditional
                m_out, v_out = layer_in.conditional_ND(self.X, full_cov=False)

            else:  # sample from q_X
                m_in = layer_in.q_X_mu
                vsqrt_in = layer_in.q_X_sqrt
                zz = tf.random_normal(tf.shape(m_in), dtype=settings.float_type)
                sampled_input = m_in + vsqrt_in * zz

                if self.layers[l-1].input_prop_dim:
                    sampled_input = tf.concat([self.X, sampled_input], 1)

                m_out, v_out = layer_in.conditional_ND(sampled_input, full_cov=False)

            if l == len(self.layers) - 1:  # final layer likelihood (i.e. data)
                E_loglik = self.likelihood.likelihood.variational_expectations
                L += tf.reduce_sum(E_loglik(m_out, v_out, self.Y))

            else:  # middle layers 'likelihoods'
                layer_out = self.layers[l + 1]
                m_next = layer_out.q_X_mu
                v_next = layer_out.q_X_sqrt ** 2
                E_loglik = layer_in.between_layer_likelihood.variational_expectations
                L += tf.reduce_sum(E_loglik(m_out, v_out, m_next, v_next))

        KL = tf.reduce_sum([layer.KL() for layer in self.layers])

        ent = tf.reduce_sum([0.5*(np.log(2*np.pi) + 1) + tf.log(layer.q_X_sqrt) for layer in self.layers[1:]])

        return L - KL + ent # minibatch not possible unless tf.gather used for q_mu

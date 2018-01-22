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

from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import conditional
from gpflow.features import InducingPoints
from gpflow.kullback_leiblers import gauss_kl
from gpflow import transforms
from gpflow import settings

from doubly_stochastic_dgp.utils import reparameterize


class SVGP_Layer(Parameterized):
    def __init__(self, kern, Z, num_outputs, mean_function):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :kern: The kernel for the layer (input_dim = D_in)
        :param q_mu: mean initialization (M, D_out)
        :param q_sqrt: sqrt of variance initialization (D_out,M,M)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function
        :return:
        """
        Parameterized.__init__(self)
        M = Z.shape[0]

        q_mu = np.zeros((M, num_outputs))
        self.q_mu = Parameter(q_mu)

        q_sqrt = np.tile(np.eye(M)[None, :, :], [num_outputs, 1, 1])
        transform = transforms.LowerTriangular(M, num_matrices=num_outputs)
        self.q_sqrt = Parameter(q_sqrt, transform=transform)

        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function

    def conditional(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """

        def single_sample_conditional(X, full_cov=False):
            mean, var = conditional(X, self.feature.Z, self.kern,
                        self.q_mu, q_sqrt=self.q_sqrt,
                        full_cov=full_cov, white=True)
            return mean + self.mean_function(X), var

        if full_cov is True:
            f = lambda a: single_sample_conditional(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = single_sample_conditional(X_flat)
            return [tf.reshape(m, [S, N, -1]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional(X, full_cov=full_cov)
        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)
        return samples, mean, var

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)

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

from doubly_stochastic_dgp.utils import PositiveExp
from doubly_stochastic_dgp.utils import normal_sample

from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import conditional, uncertain_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero, Constant
from gpflow.features import InducingPoints
from gpflow.likelihoods import Gaussian
from gpflow import settings
from gpflow import params_as_tensors
from gpflow import densities


class Layer(Parameterized):
    def __init__(self, kern, q_mu, q_sqrt, Z, mean_function, forward_propagate_inputs=False):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        Note that the we don't bother subtracting the mean from the q_mu i.e. the variational
        distribution is centered.

        The layer holds D_out independent GPs with the same kernel and inducing points.

        Note that the mean function is not identical over layers, e.g. if mean function is
        the identity and D_in = D_out

        :kern: The kernel for the layer (input_dim = D_in)
        :param q_mu: Variational mean initialization (M x D_out)
        :param q_sqrt: Variational cholesky of variance initialization (M x M x D_out)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function (e.g. Linear in the )
        :param forward_propagate_inputs: A flag to the parent model whether to concat X on the input
        :return:
        """
        Parameterized.__init__(self)
        self.q_mu, self.q_sqrt = Parameter(q_mu), Parameter(q_sqrt)
        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.forward_propagate_inputs = forward_propagate_inputs

    def conditional(self, X, full_cov=False):
        """
        Calculate the conditional of a single sample

        if full_cov is True
        mean is (N x D_out), var is (N x N x D_out)

        if full_cov is False
        mean and var are both (N x D_out)

        :param X: The input locations (N x D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (N x D_out), var (N x D_out) or (N x N x D_out)
        """
        mean, var = conditional(X, self.feature.Z, self.kern,
                                self.q_mu, q_sqrt=self.q_sqrt,
                                full_cov=full_cov, white=True)
        return mean + self.mean_function(X), var

    def multisample_conditional(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S x N x D_out), indpendent over samples

        if full_cov is True
        mean is (S x N x D_out), var is (S x N x N x D_out)

        if full_cov is False
        mean and var are both (S x N x D_out)

        :param X:  The input locations (S x N x D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S x N x D_out), var (S x N x D_out) or (S x N x N x D_out)
        """
        if full_cov is True:
            f = lambda a: self.conditional(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional(X_flat)
            return [tf.reshape(m, [S, N, -1]) for m in [mean, var]]

    def uncertain_conditional(self, X_mean, X_var, full_cov=False, full_cov_output=False):
        mean, var = uncertain_conditional(X_mean, tf.matrix_diag(X_var),  # need to make diag for now
                                          self.feature, self.kern,
                                          self.q_mu, q_sqrt=self.q_sqrt,
                                          full_cov=full_cov, white=True,
                                          full_cov_output=full_cov_output)

        if not (isinstance(self.mean_function, Zero) or isinstance(self.mean_function, Constant)):
            assert False

        return mean + self.mean_function(X_mean), var

    def multisample_uncertain_conditional(self, X_mean, X_var, full_cov=False):
        if full_cov is True:
            f = lambda a: self.uncertain_conditional(a[0], a[1], full_cov=full_cov)
            XX = [X_mean, X_var]
            mean, var = tf.map_fn(f, XX, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X_mean)[0], tf.shape(X_mean)[1], tf.shape(X_mean)[2]
            X_mean_flat = tf.reshape(X_mean, [S * N, D])
            X_var_flat = tf.reshape(X_var, [S * N, D])
            mean, var = self.uncertain_conditional(X_mean_flat, X_var_flat,
                                                   full_cov=False,
                                                   full_cov_output=False)
            return [tf.reshape(m, [S, N, -1]) for m in [mean, var]]

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)


#
# class IdentityScalingLayer(object):
#     """
#     A dummy scaling layer to use when no scaling is required.
#
#     Intended usage is
#     IdentityScalingLayer(1.) for multiplicative identity, or
#     IdentityScalingLayer(0.) for additive identity
#     """
#     def __init__(self, value):
#         self.value = tf.cast(value, dtype=settings.float_type)
#
#     def scale(self, SX, full_cov=False):
#         return self.value
#
#     def KL(self):
#         return 0.
#
#
# class ScalingLayer(Layer):
#     def __init__(self, kern, q_mu, q_sqrt, Z, avg_value, positive_transform=PositiveExp()):
#         """
#         A GP layer for applying heteroscedastic scaling e.g. for amplitude or noise variance
#
#         The layer assumes a constant mean and adds a positive transformation so the
#         outputs can be used for scaling.
#
#         Avg_value is the mean of gp pushed through the transform, i.e. the central
#         value after transforming. The layer pushes this avg_value through the inverse
#         positive transform to get the gp constant mean value
#
#         :kern: The kernel for the layer (input_dim = D_in)
#         :param q_mu: Variational mean initialization (M x D_out)
#         :param q_sqrt: Variational cholesky of variance initialization (M x M x D_out)
#         :param Z: Inducing points (M, D_in)
#         :param avg_value: The average value of the GP
#         :return:
#         """
#
#         # use the inverse positive transform to get the gp mean
#         gp_avg_value = positive_transform.backward_np(avg_value)
#
#         Layer.__init__(self, kern, q_mu, q_sqrt, Z, Constant(gp_avg_value))
#
#         self.positive_transform = positive_transform
#
#     def scale(self, SX, full_cov=False):
#         """
#         Multisample scaling values. I.e. a samples from the GP conditional pushed
#         through the transform.
#         """
#         mean, var = self.multisample_conditional(SX, full_cov=full_cov)
#         return self.positive_transform.forward(normal_sample(mean, var, full_cov=full_cov))

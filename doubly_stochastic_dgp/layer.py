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

from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import conditional, uncertain_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero, Constant
from gpflow.features import InducingPoints


class Layer(Parameterized):
    def __init__(self, kern, q_mu, q_sqrt, Z, mean_function, forward_propagate_inputs=False):
        Parameterized.__init__(self)
        self.q_mu, self.q_sqrt = Parameter(q_mu), Parameter(q_sqrt)
        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.forward_propagate_inputs = forward_propagate_inputs

    def conditional(self, X, full_cov=False):
        mean, var = conditional(X, self.feature.Z, self.kern,
                                self.q_mu, q_sqrt=self.q_sqrt,
                                full_cov=full_cov, white=True)
        return mean + self.mean_function(X), var


    def uncertain_conditional(self, X_mean, X_var, full_cov=False, full_cov_output=False):
        mean, var = uncertain_conditional(X_mean, tf.matrix_diag(X_var),  # need to make diag for now
                                          self.feature, self.kern,
                                          self.q_mu, q_sqrt=self.q_sqrt,
                                          full_cov=full_cov, white=True,
                                          full_cov_output=full_cov_output)

        if not (isinstance(self.mean_function, Zero) or isinstance(self.mean_function, Constant)):
            assert False

        return mean + self.mean_function(X_mean), var

    def multisample_conditional(self, X, full_cov=False):
        if full_cov is True:
            f = lambda a: self.conditional(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional(X_flat)
            return [tf.reshape(m, [S, N, -1]) for m in [mean, var]]

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
        return gauss_kl(self.q_mu, self.q_sqrt)












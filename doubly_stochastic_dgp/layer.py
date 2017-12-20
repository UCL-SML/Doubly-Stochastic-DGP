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
from gpflow.conditionals import conditional
from gpflow.kullback_leiblers import gauss_kl


class Layer(Parameterized):
    def __init__(self, kern, q_mu, q_sqrt, Z, mean_function, forward_propagate_inputs=False):
        Parameterized.__init__(self)
        self.q_mu, self.q_sqrt, self.Z = Parameter(q_mu), Parameter(q_sqrt), Parameter(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.forward_propagate_inputs = forward_propagate_inputs

    def conditional(self, X, full_cov=False):
        mean, var = conditional(X, self.Z, self.kern,
                                self.q_mu, q_sqrt=self.q_sqrt,
                                full_cov=full_cov, whiten=True)
        return mean + self.mean_function(X), var

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

    def KL(self):
        return gauss_kl(self.q_mu, self.q_sqrt)

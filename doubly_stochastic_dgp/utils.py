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

def normal_sample(mean, var, full_cov=False):
    if full_cov is False:
        print(settings.numerics.jitter_level)
        z = tf.random_normal(tf.shape(mean), dtype=settings.tf_float)
        return mean + z * (var + settings.numerics.jitter_level)** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.numerics.jitter_level * tf.eye(N, dtype=settings.tf_float)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)
        z = tf.random_normal([S, D, N, 1], dtype=settings.tf_float)
        f = mean + tf.matmul(chol, z)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND

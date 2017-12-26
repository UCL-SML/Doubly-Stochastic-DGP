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

import unittest

import numpy as np
import tensorflow as tf

from numpy.testing import assert_allclose

from doubly_stochastic_dgp.utils import normal_sample

class TestUtils(unittest.TestCase):
    def test(self):
        N, D = 2, 2

        m = np.random.randn(N, D)
        U = np.random.randn(N, N, D)
        S = np.einsum('nNd,mNd->nmd', U, U)
        S_diag = np.random.randn(N, D)**2

        num_samples = 100000
        m_ = np.tile(m[None, :, :], [num_samples, 1, 1])
        S_ =  np.tile(S[None, :, :, :], [num_samples, 1, 1, 1])
        S_diag_ =  np.tile(S_diag[None, :, :], [num_samples, 1, 1])

        with tf.Session() as sess:
            samples_diag = sess.run(normal_sample(m_, S_diag_))
            samples = sess.run(normal_sample(m_, S_, full_cov=True))

        assert_allclose(np.average(samples, 0), m, atol=0.05)

        cov = np.array([np.cov(s) for s in np.transpose(samples, [2, 1, 0])])
        cov = np.transpose(cov, [1, 2, 0])
        assert_allclose(cov, S, atol=0.05)

        assert_allclose(np.average(samples_diag, 0), m, atol=0.05)
        assert_allclose(np.std(samples_diag, 0)**2, S_diag, atol=0.05)


if __name__ == '__main__':
    unittest.main()


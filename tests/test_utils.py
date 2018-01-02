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
from doubly_stochastic_dgp.utils import PositiveExp, PositiveSoftplus

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


class TestForwardBackward(unittest.TestCase):
    def setUp(self):
        N, D = 3, 2
        self.x = np.random.randn(N, D)
        self.y = np.random.randn(N, D)**2
        self.pos_exp = PositiveExp()
        self.pos_softplus = PositiveSoftplus()

    def test_tf_np(self):
        with tf.Session() as sess:
            for f in self.pos_exp, self.pos_softplus:
                y_tf = sess.run(f.forward(tf.cast(self.x, tf.float64)))
                y_np = f.forward_np(self.x)
                assert_allclose(y_tf, y_np)

                x_tf = sess.run(f.backward(tf.cast(self.y, tf.float64)))
                x_np = f.backward_np(self.y)
                assert_allclose(x_tf, x_np)

    def test_postive(self):
        with tf.Session() as sess:
            for f in self.pos_exp, self.pos_softplus:
                y_tf = sess.run(f.forward(tf.cast(self.x, tf.float64)))
                assert np.all(y_tf > 0.)

    def test_inverse(self):
        with tf.Session() as sess:
            for f in self.pos_exp, self.pos_softplus:
                x_tf = sess.run(f.backward(f.forward(tf.cast(self.x, tf.float64))))
                assert_allclose(x_tf, self.x)
                y_tf = sess.run(f.forward(f.backward(tf.cast(self.y, tf.float64))))
                assert_allclose(y_tf, self.y)

    def test_extreme_for_softplus(self):
        with tf.Session() as sess:
            f = self.pos_softplus
            x = np.linspace(-5, 1000, 100)
            y = np.linspace(1e-5, 1000, 100)

            x_tf = sess.run(f.backward(f.forward(tf.cast(x, tf.float64))))
            assert_allclose(x_tf, x)
            y_tf = sess.run(f.forward(f.backward(tf.cast(y, tf.float64))))
            assert_allclose(y_tf, y)

            x_neg = np.linspace(-1000, 0)
            y_neg = sess.run(f.forward(tf.cast(x_neg, tf.float64)))
            assert(not np.any(np.isnan(y_neg)))
            assert np.all(y_neg>0.)


if __name__ == '__main__':
    unittest.main()

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

from gpflow import autoflow, params_as_tensors
from gpflow.likelihoods import *
from gpflow.models import Model
from gpflow import settings

from doubly_stochastic_dgp.utils import normal_sample
from doubly_stochastic_dgp.utils import PositiveExp, PositiveSoftplus
from doubly_stochastic_dgp.utils import LikelihoodWrapper




class LikelihoodTester(Model):
    def __init__(self, likelihood):
        Model.__init__(self)
        self.wrapped_likelihood = LikelihoodWrapper(likelihood)
        self.likelihood = likelihood

    def _build_likelihood(self):
        return tf.cast(0., dtype=settings.float_type)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]), (settings.float_type, [None, None]))
    def logp1(self, F, Y):
        return self.wrapped_likelihood.logp(F, Y)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]), (settings.float_type, [None, None]))
    def logp2(self, F, Y):
        f = lambda a: self.likelihood.logp(a, Y)
        return tf.stack(tf.map_fn(f, F, dtype=settings.float_type))

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]))
    def conditional_mean1(self, F):
        return self.wrapped_likelihood.conditional_mean(F)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]))
    def conditional_mean2(self, F):
        f = lambda a: self.likelihood.conditional_mean(a)
        return tf.stack(tf.map_fn(f, F, dtype=settings.float_type))

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]))
    def conditional_variance1(self, F):
        return self.wrapped_likelihood.conditional_variance(F)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]))
    def conditional_variance2(self, F):
        f = lambda a: self.likelihood.conditional_variance(a)
        return tf.stack(tf.map_fn(f, F, dtype=settings.float_type))

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]))
    def predict_mean_and_var1(self, Fmu, Fvar):
        return self.wrapped_likelihood.predict_mean_and_var(Fmu, Fvar)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]))
    def predict_mean_and_var2(self,  Fmu, Fvar):
        f = lambda a: list(self.likelihood.predict_mean_and_var(a[0], a[1]))
        m, v = tf.map_fn(f, [Fmu, Fvar], dtype=[settings.float_type, settings.float_type])
        return tf.stack(m), tf.stack(v)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]),
              (settings.float_type, [None, None]))
    def predict_density1(self, Fmu, Fvar, Y):
        return self.wrapped_likelihood.predict_density(Fmu, Fvar, Y)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]),
              (settings.float_type, [None, None]))
    def predict_density2(self,  Fmu, Fvar, Y):
        f = lambda a: self.likelihood.predict_density(a[0], a[1], Y)
        return tf.stack(tf.map_fn(f, [Fmu, Fvar], dtype=settings.float_type))

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]),
              (settings.float_type, [None, None]))
    def variational_expectations1(self, Fmu, Fvar, Y):
        return self.wrapped_likelihood.variational_expectations(Fmu, Fvar, Y)

    @params_as_tensors
    @autoflow((settings.float_type, [None, None, None]),
              (settings.float_type, [None, None, None]),
              (settings.float_type, [None, None]))
    def variational_expectations2(self,  Fmu, Fvar, Y):
        f = lambda a: self.likelihood.variational_expectations(a[0], a[1], Y)
        return tf.stack(tf.map_fn(f, [Fmu, Fvar], dtype=settings.float_type))


class TestLikelihoodWrapper(unittest.TestCase):
    def setUp(self):
        S, N, D = 5, 4, 3
        self.Fmu = np.random.randn(S, N, D)
        self.Fvar = np.random.randn(S, N, D)**2
        self.Y = np.ones((N, D))

    def run_tests(self, likelihood, Fmu, Fvar, Y):
        l = LikelihoodTester(likelihood)
        assert_allclose(l.logp1(Fmu, Y), l.logp2(Fmu, Y))
        assert_allclose(l.conditional_mean1(Fmu), l.conditional_mean2(Fmu))
        assert_allclose(l.conditional_variance1(Fmu), l.conditional_variance2(Fmu))

        m1, v1 = l.predict_mean_and_var1(Fmu, Fvar)
        m2, v2 = l.predict_mean_and_var2(Fmu, Fvar)
        assert_allclose(m1, m2)
        assert_allclose(v1, v2)

        assert_allclose(l.predict_density1(Fmu, Fvar, Y),
                        l.predict_density2(Fmu, Fvar, Y))

        assert_allclose(l.variational_expectations1(Fmu, Fvar, Y),
                        l.variational_expectations2(Fmu, Fvar, Y))

    def test_gaussian(self):
        self.run_tests(Gaussian(), self.Fmu, self.Fvar, self.Y)

    def test_bernoulli(self):
        self.run_tests(Bernoulli(), self.Fmu, self.Fvar, self.Y)

    # def test_multiclass(self):
    #     K = self.Fmu.shape[2]
    #     Y = np.ones((self.Fmu.shape[1], K))
    #     self.run_tests(MultiClass(K), self.Fmu, self.Fvar, Y)

    def test_exponential(self):
        self.run_tests(Exponential(), self.Fmu, self.Fvar, self.Y)

    def test_poisson(self):
        self.run_tests(Poisson(), self.Fmu, self.Fvar, self.Y)

    def test_studentT(self):
        self.run_tests(StudentT(), self.Fmu, self.Fvar, self.Y)

    def test_studentT(self):
        self.run_tests(Gamma(), self.Fmu, self.Fvar, self.Y)

    def test_beta(self):
        self.run_tests(Beta(), self.Fmu, self.Fvar, self.Y)

# class TestSampling(unittest.TestCase):
#     def test(self):
#         N, D = 2, 2
#
#         m = np.random.randn(N, D)
#         U = np.random.randn(N, N, D)
#         S = np.einsum('nNd,mNd->nmd', U, U)
#         S_diag = np.random.randn(N, D)**2
#
#         num_samples = 100000
#         m_ = np.tile(m[None, :, :], [num_samples, 1, 1])
#         S_ =  np.tile(S[None, :, :, :], [num_samples, 1, 1, 1])
#         S_diag_ =  np.tile(S_diag[None, :, :], [num_samples, 1, 1])
#
#         with tf.Session() as sess:
#             samples_diag = sess.run(normal_sample(m_, S_diag_))
#             samples = sess.run(normal_sample(m_, S_, full_cov=True))
#
#         assert_allclose(np.average(samples, 0), m, atol=0.05)
#
#         cov = np.array([np.cov(s) for s in np.transpose(samples, [2, 1, 0])])
#         cov = np.transpose(cov, [1, 2, 0])
#         assert_allclose(cov, S, atol=0.05)
#
#         assert_allclose(np.average(samples_diag, 0), m, atol=0.05)
#         assert_allclose(np.std(samples_diag, 0)**2, S_diag, atol=0.05)
#
#
# class TestForwardBackward(unittest.TestCase):
#     def setUp(self):
#         N, D = 3, 2
#         self.x = np.random.randn(N, D)
#         self.y = np.random.randn(N, D)**2
#         self.pos_exp = PositiveExp()
#         self.pos_softplus = PositiveSoftplus()
#
#     def test_tf_np(self):
#         with tf.Session() as sess:
#             for f in self.pos_exp, self.pos_softplus:
#                 y_tf = sess.run(f.forward(tf.cast(self.x, tf.float64)))
#                 y_np = f.forward_np(self.x)
#                 assert_allclose(y_tf, y_np)
#
#                 x_tf = sess.run(f.backward(tf.cast(self.y, tf.float64)))
#                 x_np = f.backward_np(self.y)
#                 assert_allclose(x_tf, x_np)
#
#     def test_postive(self):
#         with tf.Session() as sess:
#             for f in self.pos_exp, self.pos_softplus:
#                 y_tf = sess.run(f.forward(tf.cast(self.x, tf.float64)))
#                 assert np.all(y_tf > 0.)
#
#     def test_inverse(self):
#         with tf.Session() as sess:
#             for f in self.pos_exp, self.pos_softplus:
#                 x_tf = sess.run(f.backward(f.forward(tf.cast(self.x, tf.float64))))
#                 assert_allclose(x_tf, self.x)
#                 y_tf = sess.run(f.forward(f.backward(tf.cast(self.y, tf.float64))))
#                 assert_allclose(y_tf, self.y)
#
#     def test_extreme_for_softplus(self):
#         with tf.Session() as sess:
#             f = self.pos_softplus
#             x = np.linspace(-5, 1000, 100)
#             y = np.linspace(1e-5, 1000, 100)
#
#             x_tf = sess.run(f.backward(f.forward(tf.cast(x, tf.float64))))
#             assert_allclose(x_tf, x)
#             y_tf = sess.run(f.forward(f.backward(tf.cast(y, tf.float64))))
#             assert_allclose(y_tf, y)
#
#             x_neg = np.linspace(-1000, 0)
#             y_neg = sess.run(f.forward(tf.cast(x_neg, tf.float64)))
#             assert(not np.any(np.isnan(y_neg)))
#             assert np.all(y_neg>0.)


if __name__ == '__main__':
    unittest.main()

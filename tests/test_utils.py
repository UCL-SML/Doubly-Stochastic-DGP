# # Copyright 2017 Hugh Salimbeni
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# import unittest
#
# import numpy as np
# import tensorflow as tf
#
# from numpy.testing import assert_allclose
#
# from gpflow import autoflow, params_as_tensors
# from gpflow.likelihoods import *
# from gpflow.models import Model
# from gpflow import settings
#
# from doubly_stochastic_dgp.utils import reparameterize
# from doubly_stochastic_dgp.utils import BroadcastingLikelihood
#
#
# class LikelihoodTester(Model):
#     def __init__(self, likelihood):
#         Model.__init__(self)
#         self.wrapped_likelihood = BroadcastingLikelihood(likelihood)
#         self.likelihood = likelihood
#
#     def _build_likelihood(self):
#         return tf.cast(0., dtype=settings.float_type)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]), (settings.float_type, [None, None]))
#     def logp1(self, F, Y):
#         return self.wrapped_likelihood.logp(F, Y)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]), (settings.float_type, [None, None]))
#     def logp2(self, F, Y):
#         f = lambda a: self.likelihood.logp(a, Y)
#         return tf.stack(tf.map_fn(f, F, dtype=settings.float_type))
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]))
#     def conditional_mean1(self, F):
#         return self.wrapped_likelihood.conditional_mean(F)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]))
#     def conditional_mean2(self, F):
#         f = lambda a: tf.cast(self.likelihood.conditional_mean(a), dtype=settings.float_type)
#         return tf.stack(tf.map_fn(f, F, dtype=settings.float_type))
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]))
#     def conditional_variance1(self, F):
#         return self.wrapped_likelihood.conditional_variance(F)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]))
#     def conditional_variance2(self, F):
#         f = lambda a: tf.cast(self.likelihood.conditional_variance(a), dtype=settings.float_type)
#         return tf.stack(tf.map_fn(f, F, dtype=settings.float_type))
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None, None]))
#     def predict_mean_and_var1(self, Fmu, Fvar):
#         return self.wrapped_likelihood.predict_mean_and_var(Fmu, Fvar)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None, None]))
#     def predict_mean_and_var2(self,  Fmu, Fvar):
#         f = lambda a: list(self.likelihood.predict_mean_and_var(a[0], a[1]))
#         m, v = tf.map_fn(f, [Fmu, Fvar], dtype=[settings.float_type, settings.float_type])
#         return tf.stack(m), tf.stack(v)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None]))
#     def predict_density1(self, Fmu, Fvar, Y):
#         return self.wrapped_likelihood.predict_density(Fmu, Fvar, Y)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None]))
#     def predict_density2(self,  Fmu, Fvar, Y):
#         f = lambda a: self.likelihood.predict_density(a[0], a[1], Y)
#         return tf.stack(tf.map_fn(f, [Fmu, Fvar], dtype=settings.float_type))
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None]))
#     def variational_expectations1(self, Fmu, Fvar, Y):
#         return self.wrapped_likelihood.variational_expectations(Fmu, Fvar, Y)
#
#     @params_as_tensors
#     @autoflow((settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None, None]),
#               (settings.float_type, [None, None]))
#     def variational_expectations2(self,  Fmu, Fvar, Y):
#         f = lambda a: self.likelihood.variational_expectations(a[0], a[1], Y)
#         return tf.stack(tf.map_fn(f, [Fmu, Fvar], dtype=settings.float_type))
#
#
# class TestLikelihoodWrapper(unittest.TestCase):
#     def setUp(self):
#         S, N, D = 5, 4, 3
#         self.Fmu = np.random.randn(S, N, D)
#         self.Fvar = np.random.randn(S, N, D)**2
#         self.N, self.D = N, D
#
#     def run_tests(self, likelihood, Fmu, Fvar, Y):
#         l = LikelihoodTester(likelihood)
#         assert_allclose(l.logp1(Fmu, Y), l.logp2(Fmu, Y))
#         assert_allclose(l.conditional_mean1(Fmu), l.conditional_mean2(Fmu))
#         assert_allclose(l.conditional_variance1(Fmu), l.conditional_variance2(Fmu))
#
#         m1, v1 = l.predict_mean_and_var1(Fmu, Fvar)
#         m2, v2 = l.predict_mean_and_var2(Fmu, Fvar)
#         assert_allclose(m1, m2)
#         assert_allclose(v1, v2)
#
#         assert_allclose(l.predict_density1(Fmu, Fvar, Y),
#                         l.predict_density2(Fmu, Fvar, Y))
#
#         assert_allclose(l.variational_expectations1(Fmu, Fvar, Y),
#                         l.variational_expectations2(Fmu, Fvar, Y))
#
#     def test_gaussian(self):
#         self.run_tests(Gaussian(), self.Fmu, self.Fvar, np.random.randn(self.N, self.D))
#
#     def test_bernoulli(self):
#         Y = np.random.choice([0., 1.], self.N * self.D).reshape(self.N, self.D)
#         self.run_tests(Bernoulli(), self.Fmu, self.Fvar, Y)
#
#     def test_multiclass(self):
#         K = self.Fmu.shape[2]
#         Y = np.random.choice(np.arange(K).astype(float), self.Fmu.shape[1]).reshape(-1, 1)
#         self.run_tests(MultiClass(K), self.Fmu, self.Fvar, Y)
#
#     def test_exponential(self):
#         Y = np.random.randn(self.N, self.D)**2
#         self.run_tests(Exponential(), self.Fmu, self.Fvar, Y)
#
#     def test_poisson(self):
#         Y = np.floor(np.random.randn(self.N, self.D)**2).astype(float)
#         self.run_tests(Poisson(), self.Fmu, self.Fvar, Y)
#
#     def test_studentT(self):
#         Y = np.random.randn(self.N, self.D)
#         self.run_tests(StudentT(), self.Fmu, self.Fvar, Y)
#
#     def test_gamma(self):
#         Y = np.random.randn(self.N, self.D)**2
#         self.run_tests(Gamma(), self.Fmu, self.Fvar, Y)
#
#     def test_beta(self):
#         Y = np.random.randn(self.N, self.D)
#         Y = 1/(1+np.exp(-Y))
#         self.run_tests(Beta(), self.Fmu, self.Fvar, Y)
#
#     def test_ordinal(self):
#         Y = np.random.choice(range(4), self.N*self.D).reshape(self.N, self.D).astype(float)
#         self.run_tests(Ordinal(np.linspace(-2, 2, 4)), self.Fmu, self.Fvar, Y)
#
#
# class TestReparameterize(unittest.TestCase):
#     def testReparameterizeDiag(self):
#         S, N, D = 4, 3, 2
#         mean = np.random.randn(S, N, D)
#         var = np.random.randn(S, N, D)**2
#         z = np.random.randn(S, N, D)
#         f = mean + z * (var + 1e-6)**0.5
#         with tf.Session() as sess:
#             assert_allclose(f, sess.run(reparameterize(tf.identity(mean), var, z)))
#
#     def testReparameterizeFullCov(self):
#         S, N, D = 4, 3, 2
#
#         mean = np.random.randn(S, N, D)
#         U = np.random.randn(S, N, N, D)
#         var = np.einsum('SnNd,SmNd->Snmd', U, U) + np.eye(N)[None, :, :, None] * 1e-6
#
#         var_flat = np.reshape(np.transpose(var, [0, 3, 1, 2]), [S*D, N, N])
#         L_flat = np.linalg.cholesky(var_flat + np.eye(N)[None, :, :] * 1e-6)
#         L = np.transpose(np.reshape(L_flat, [S, D, N, N]), [0, 2, 3, 1])
#
#         z = np.random.randn(S, N, D)
#         f = mean + np.einsum('SnNd,SNd->Snd', L, z)
#
#         with tf.Session() as sess:
#             assert_allclose(f, sess.run(reparameterize(tf.identity(mean), var, z,
#                                                        full_cov=True)))
#
#
# if __name__ == '__main__':
#     unittest.main()

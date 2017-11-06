# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:46:18 2017

@author: hrs13
"""

import unittest
import numpy as np

from gpflow._settings import settings
settings.numerics.jitter_level=1e-15

from gpflow.svgp import SVGP
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF

from doubly_stochastic_dgp.dgp import DGP


class InitializationsTests(unittest.TestCase):
    def setUp(self):
        N, D, D_Y = 3, 1, 1
        self.X = np.random.randn(N, D)
        self.Y = np.random.randn(N, D_Y)
        self.lik_final = Gaussian()
        self.lik_final.variance = 1e-1
        self.kern_final = RBF(D, lengthscales=float(D)**0.5)
        self.svgp = SVGP(self.X, self.Y, self.kern_final, self.lik_final, self.X)

        self.q_mu = np.random.randn(N, 1)

        # self.q_sqrt = np.eye(N)[:, :, None] * np.ones((1, 1, D_Y))
        u = np.random.randn(N, N, D)
        self.q_sqrt = np.einsum('nmd,Nmd->nNd', u, u)

        self.svgp.q_mu = self.q_mu
        self.svgp.q_sqrt = self.q_sqrt

    def test_initializations_linear_mf(self):
        D = self.X.shape[1]
        kerns = [RBF(D, lengthscales=10), self.kern_final]
        m_dgp = DGP(self.X, self.Y, self.X, kerns, self.lik_final)

        for layer in m_dgp.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-12

        m_dgp.layers[-1].q_mu = self.q_mu
        m_dgp.layers[-1].q_sqrt = self.q_sqrt

        mean_svgp, var_svgp = self.svgp.predict_f(self.X)

        means_dgp, vars_dgp = m_dgp.predict_f(self.X, 3)

        for mean_dgp, var_dgp in zip(means_dgp, vars_dgp):
            assert np.allclose(mean_dgp, mean_svgp)
            assert np.allclose(var_dgp, var_svgp)

    def test_initializations_zero_mf(self):
        D = self.X.shape[1]
        kerns = [RBF(D, lengthscales=10), self.kern_final]
        m_dgp = DGP(self.X, self.Y, self.X, kerns, self.lik_final,
                    linear_mean_functions=False)

        for layer in m_dgp.layers[:-1]:
            K = layer.kern.compute_K_symm(layer.Z.value)
            L = np.linalg.cholesky(K + np.eye(self.X.shape[0])*1e-12)
            layer.q_mu = np.linalg.solve(L, self.X)
            layer.q_sqrt = layer.q_sqrt.value * 1e-12

        m_dgp.layers[-1].q_mu = self.q_mu
        m_dgp.layers[-1].q_sqrt = self.q_sqrt

        mean_svgp, var_svgp = self.svgp.predict_f(self.X)

        means_dgp, vars_dgp = m_dgp.predict_f(self.X, 3)

        for mean_dgp, var_dgp in zip(means_dgp, vars_dgp):
            assert np.allclose(mean_dgp, mean_svgp)
            assert np.allclose(var_dgp, var_svgp)

    def test_initalizations_forward_prop(self):
        D = self.X.shape[1]
        final_kernel = RBF(2*D, lengthscales=self.kern_final.lengthscales.value)
        kerns = [RBF(2*D, lengthscales=10), final_kernel]
        m_dgp = DGP(self.X, self.Y, self.X, kerns, self.lik_final,
                    forward_propagate_inputs=True,
                    linear_mean_functions=False)

        for layer in m_dgp.layers:
            layer.Z = np.concatenate([self.X, 0*self.X], 1)

        for layer in m_dgp.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-18

        m_dgp.layers[-1].q_mu = self.q_mu
        m_dgp.layers[-1].q_sqrt = self.q_sqrt

        mean_svgp, var_svgp = self.svgp.predict_f(self.X)

        means_dgp, vars_dgp = m_dgp.predict_f(self.X, 3)

        for mean_dgp, var_dgp in zip(means_dgp, vars_dgp):
            assert np.allclose(mean_dgp, mean_svgp)
            assert np.allclose(var_dgp, var_svgp)

    def test_initalizations_forward_prop_with_linear(self):
        D = self.X.shape[1]
        final_kernel = RBF(2 * D, lengthscales=self.kern_final.lengthscales.value)
        kerns = [RBF(2 * D, lengthscales=10), final_kernel]
        m_dgp = DGP(self.X, self.Y, self.X, kerns, self.lik_final,
                    forward_propagate_inputs=True,
                    linear_mean_functions=True)

        for layer in m_dgp.layers:
            layer.Z = np.concatenate([self.X, 0*self.X], 1)

        for layer in m_dgp.layers[:-1]:
            D = self.X.shape[1]
            W_new = np.zeros((2*D, D))
            layer.mean_function.A = W_new
            layer.q_sqrt = layer.q_sqrt.value * 1e-18

        m_dgp.layers[-1].q_mu = self.q_mu
        m_dgp.layers[-1].q_sqrt = self.q_sqrt

        mean_svgp, var_svgp = self.svgp.predict_f(self.X)

        means_dgp, vars_dgp = m_dgp.predict_f(self.X, 3)
        for mean_dgp, var_dgp in zip(means_dgp, vars_dgp):
            assert np.allclose(mean_dgp, mean_svgp)
            assert np.allclose(var_dgp, var_svgp)


# class DGPTests(unittest.TestCase):
#     def test_vs_single_layer_GP(self):
#         N, D_X, D_Y, M, Ns = 5, 4, 3, 5, 5
#
#         X = np.random.randn(N, D_X)
#         Y = np.random.randn(N, D_Y)
#         Z = np.random.randn(M, D_X)
#
#         Xs = np.random.randn(Ns, D_X)
#         Ys = np.random.randn(Ns, D_Y)
#
#         noise = np.random.gamma([1, ])
#         ls = np.random.gamma([D_X, ])
#         s = np.random.gamma([D_Y, ])
#         q_mu = np.random.randn(M, D_Y)
#         q_sqrt = np.random.randn(M, M, D_Y)
#
#         m_svgp = SVGP(X, Y, RBF(D_X), Gaussian(), Z,
#                       q_diag=False, whiten=True)
#
#         m_svgp.kern.lengthscales = ls
#         m_svgp.kern.variance = s
#         m_svgp.likelihood.variance = noise
#         m_svgp.q_mu = q_mu
#         m_svgp.q_sqrt = q_sqrt
#
#         def make_dgp_as_sgp(kernels):
#             m_dgp = DGP(X, Y, Z, kernels, Gaussian())
#
#             #set final layer to sgp
#             m_dgp.layers[-1].kern.lengthscales = ls
#             m_dgp.layers[-1].kern.variance = s
#             m_dgp.likelihood.variance = noise
#             m_dgp.layers[-1].q_mu = q_mu
#             m_dgp.layers[-1].q_sqrt = q_sqrt
#
#             # set other layers to identity
#             for layer in m_dgp.layers[:-1]:
# #                1e-6 gives errors of 1e-3, so need to set right down
#                 layer.kern.variance.transform._lower = 1e-18
#                 layer.kern.variance = 1e-18
#
#             return m_dgp
#
#         m_dgp_1 = make_dgp_as_sgp([RBF(D_X), ])
#         m_dgp_2 = make_dgp_as_sgp([RBF(D_X), RBF(D_X)])
#         m_dgp_3 = make_dgp_as_sgp([RBF(D_X), RBF(D_X), RBF(D_X)])
#
#         preds_svgp = m_svgp.predict_f(Xs)
#         preds_dgp_1 = [mv[0] for mv in m_dgp_1.predict_f(Xs, 1)]
#         preds_dgp_2 = [mv[0] for mv in m_dgp_2.predict_f(Xs, 1)]
#         preds_dgp_3 = [mv[0] for mv in m_dgp_3.predict_f(Xs, 1)]
#
#         assert np.allclose(preds_svgp, preds_dgp_1)
#         assert np.allclose(preds_svgp, preds_dgp_2)
#         assert np.allclose(preds_svgp, preds_dgp_3)
#
#         density_gp = m_svgp.predict_density(Xs, Ys)
#
#         density_dgp_1 = m_dgp_1.predict_density(Xs, Ys, 2)
#         density_dgp_2 = m_dgp_2.predict_density(Xs, Ys, 2)
#         density_dgp_3 = m_dgp_3.predict_density(Xs, Ys, 2)
#
#         assert np.allclose(density_dgp_1, density_gp)
#         assert np.allclose(density_dgp_2, density_gp)
#         assert np.allclose(density_dgp_3, density_gp)


if __name__ == '__main__':
    unittest.main()

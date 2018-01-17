import unittest
import numpy as np

from numpy.testing import assert_allclose

from gpflow import settings, session_manager
from gpflow.models.svgp import SVGP
from gpflow.kernels import RBF
from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass

from doubly_stochastic_dgp.layer_initializations import init_layers_input_propagation
from doubly_stochastic_dgp.layer_initializations import init_layers_linear_mean_functions
from doubly_stochastic_dgp.dgp import DGP

np.random.seed(0)


class TestVsSingleLayer(unittest.TestCase):
    def setUp(self):
        Ns, N, D_X, D_Y = 5, 4, 3, 2

        self.X = np.random.uniform(size=(N, D_X))
        self.Xs = np.random.uniform(size=(Ns, D_X))
        self.q_mu = np.random.randn(N, D_Y)
        self.q_sqrt = np.random.randn(D_Y, N, N)
        self.D_Y = D_Y

    def test_gaussian_linear(self):
        lik = Gaussian()
        lik.variance = 0.01
        N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        Y = np.random.randn(N, D_Y)
        Ys = np.random.randn(Ns, D_Y)
        self.compare_to_single_layer(Y, Ys, lik,
                                     init_layers_linear_mean_functions)

    def test_gaussian_input_prop(self):
        lik = Gaussian()
        lik.variance = 0.01

        N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        Y = np.random.randn(N, D_Y)
        Ys = np.random.randn(Ns, D_Y)
        self.compare_to_single_layer(Y, Ys, lik,
                                     init_layers_input_propagation)

    def test_bernoulli_linear(self):
        lik = Bernoulli()
        N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        Y = np.random.choice([-1., 1.], N*D_Y).reshape(N, D_Y)
        Ys = np.random.choice([-1., 1.], Ns*D_Y).reshape(Ns, D_Y)
        self.compare_to_single_layer(Y, Ys, lik,
                                     init_layers_linear_mean_functions)

    def compare_to_single_layer(self, Y, Ys, lik, init_method):
        kern = Matern52(self.X.shape[1], lengthscales=0.1)

        m_svgp = SVGP(self.X, Y, kern, lik, Z=self.X)
        m_svgp.q_mu = self.q_mu
        m_svgp.q_sqrt = self.q_sqrt

        L_svgp = m_svgp.compute_log_likelihood()
        mean_svgp, var_svgp = m_svgp.predict_y(self.Xs)
        test_lik_svgp = m_svgp.predict_density(self.Xs, Ys)
        pred_m_svgp, pred_v_svgp = m_svgp.predict_f(self.Xs)
        pred_mfull_svgp, pred_vfull_svgp = m_svgp.predict_f_full_cov(self.Xs)

        m_dgp = DGP(self.X, Y, self.X, [kern], lik, init_layers=init_method)
        m_dgp.layers[0].q_mu = self.q_mu
        m_dgp.layers[0].q_sqrt = self.q_sqrt

        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_y(self.Xs, 1)
        test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)

        pred_m_dgp, pred_v_dgp = m_dgp.predict_f(self.Xs, 1)
        pred_mfull_dgp, pred_vfull_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)

        assert_allclose(L_svgp, L_dgp)

        assert_allclose(mean_svgp, mean_dgp[0])
        assert_allclose(var_svgp, var_dgp[0])
        assert_allclose(test_lik_svgp, test_lik_dgp)

        assert_allclose(pred_m_dgp, pred_m_svgp)
        assert_allclose(pred_v_dgp, pred_v_svgp)
        assert_allclose(pred_mfull_dgp, pred_mfull_svgp)
        assert_allclose(pred_vfull_dgp, pred_vfull_svgp)



class TestVsSingleLayer(unittest.TestCase):
    def setUp(self):
        self.Ns, self.N, self.D_X, self.D_Y = 5, 4, 2, 3

        self.X = np.random.uniform(low=-5, high=5, size=(self.N, self.D_X))
        self.Xs = np.random.uniform(low=-5, high=5, size=(self.Ns, self.D_X))

        self.q_mu = np.random.randn(self.N, self.D_Y)
        self.q_sqrt = np.random.randn(self.D_Y, self.N, self.N)

        self.Y = np.random.randn(self.N, self.D_Y)
        self.Ys = np.random.randn(self.Ns, self.D_Y)

    def test_2_linear_mean_gaussian_RBF_analytic_expectations(self):
        self.compare_linear_mean(2, self.Y, self.Ys, Gaussian(), RBF, True)

    def test_3_linear_mean_gaussian_RBF_analytic_expectations(self):
        self.compare_linear_mean(3, self.Y, self.Ys, Gaussian(), RBF, True)
#
#     def test_2_linear_mean_gaussian_Matern_analytic_expectations(self):
#         self.compare_linear_mean(2, self.Y, self.Ys, Gaussian(), Matern52, True)
#
#     def test_3_linear_mean_gaussian_Matern_analytic_expectations(self):
#         self.compare_linear_mean(3, self.Y, self.Ys, Gaussian(), Matern52, True)
#
#     def test_2_linear_mean_gaussian(self):
#         self.compare_linear_mean(2, self.Y, self.Ys, Gaussian(), RBF, False)
#
#     def test_3_linear_mean_gaussian(self):
#         self.compare_linear_mean(3, self.Y, self.Ys, Gaussian(), RBF, False)
#
#     def test_2_linear_mean_bernoulli_RBF_analytic_expectations(self):
#         self.compare_linear_mean(2, self.Y, self.Ys, Bernoulli(), RBF, True)
#
#     def test_3_linear_mean_bernoulli_RBF_analytic_expectations(self):
#         self.compare_linear_mean(3, self.Y, self.Ys, Bernoulli(), RBF, True)
#
#     def test_2_linear_mean_bernoulli_Matern_analytic_expectations(self):
#         self.compare_linear_mean(2, self.Y, self.Ys, Bernoulli(), Matern52, True)
#
#     def test_3_linear_mean_bernoulli_Matern_analytic_expectations(self):
#         self.compare_linear_mean(3, self.Y, self.Ys, Bernoulli(), Matern52, True)
#
#     def test_2_linear_mean_bernoulli(self):
#         self.compare_linear_mean(2, self.Y, self.Ys, Bernoulli(), RBF, False)
#
#     def test_3_linear_mean_bernoulli(self):
#         self.compare_linear_mean(3, self.Y, self.Ys, Bernoulli(), RBF, False)
#
    def compare_linear_mean(self, L, Y, Ys, lik, Kern, analytic_exp):

        m_svgp = SVGP(self.X, Y, Kern(self.D_X), lik, Z=self.X)
        m_svgp.q_mu = self.q_mu
        m_svgp.q_sqrt = self.q_sqrt

        L_svgp = m_svgp.compute_log_likelihood()
        mean_svgp, var_svgp = m_svgp.predict_f(self.Xs)
        test_lik_svgp = m_svgp.predict_density(self.Xs, Ys)
        pred_m_svgp, pred_v_svgp = m_svgp.predict_f(self.Xs)
        pred_mfull_svgp, pred_vfull_svgp = m_svgp.predict_f_full_cov(self.Xs)

        init_method = init_layers_linear_mean_functions

        kerns = []
        for l in range(L):
            kerns.append(Kern(self.D_X))

        m_dgp = DGP(self.X, Y, self.X, kerns, lik,
                    init_layers=init_method)#,
                    # analytic_final_expectations=analytic_exp)

        for layer in m_dgp.layers[:-1]:
            layer.kern.variance = 1e-12

        m_dgp.layers[-1].q_mu = self.q_mu
        m_dgp.layers[-1].q_sqrt = self.q_sqrt

        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_f(self.Xs, 1)
        test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)

        pred_m_dgp, pred_v_dgp = m_dgp.predict_f(self.Xs, 1)
        pred_mfull_dgp, pred_vfull_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)

        assert_allclose(L_svgp, L_dgp, atol=1e-3, rtol=1e-3)

        assert_allclose(mean_svgp, mean_dgp[0], atol=1e-3, rtol=1e-3)
        assert_allclose(var_svgp, var_dgp[0],atol=1e-3, rtol=1e-3)
        assert_allclose(test_lik_svgp, test_lik_dgp, atol=1e-3, rtol=1e-3)

        assert_allclose(pred_m_dgp[0], pred_m_svgp, atol=1e-3, rtol=1e-3)
        assert_allclose(pred_v_dgp[0], pred_v_svgp, atol=1e-3, rtol=1e-3)
        assert_allclose(pred_mfull_dgp[0], pred_mfull_svgp, atol=1e-3, rtol=1e-3)
        assert_allclose(pred_vfull_dgp[0], pred_vfull_svgp, atol=1e-3, rtol=1e-3)

#
#     def test_2_input_prop_gaussian_RBF_analytic_expectations(self):
#         self.compare_input_prop(2, self.Y, self.Ys, Gaussian(), RBF, True)
#
#     def test_3_input_prop_gaussian_RBF_analytic_expectations(self):
#         self.compare_input_prop(3, self.Y, self.Ys, Gaussian(), RBF, True)
#
#     def test_2_input_prop_gaussian_Matern_analytic_expectations(self):
#         self.compare_input_prop(2, self.Y, self.Ys, Gaussian(), Matern52, True)
#
#     def test_3_input_prop_gaussian_Matern_analytic_expectations(self):
#         self.compare_input_prop(3, self.Y, self.Ys, Gaussian(), Matern52, True)
#
#     def test_2_input_prop_gaussian(self):
#         self.compare_input_prop(2, self.Y, self.Ys, Gaussian(), RBF, False)
#
#     def test_3_input_prop_gaussian(self):
#         self.compare_input_prop(3, self.Y, self.Ys, Gaussian(), RBF, False)
#
#     def test_2_input_prop_bernoulli_RBF_analytic_expectations(self):
#         self.compare_input_prop(2, self.Y, self.Ys, Bernoulli(), RBF, True)
#
#     def test_3_input_prop_bernoulli_RBF_analytic_expectations(self):
#         self.compare_input_prop(3, self.Y, self.Ys, Bernoulli(), RBF, True)
#
#     def test_2_input_prop_bernoulli_Matern_analytic_expectations(self):
#         self.compare_input_prop(2, self.Y, self.Ys, Bernoulli(), Matern52, True)
#
#     def test_3_input_prop__bernoulli_Matern_analytic_expectations(self):
#         self.compare_input_prop(3, self.Y, self.Ys, Bernoulli(), Matern52, True)
#
#     def test_2_input_prop__bernoulli(self):
#         self.compare_input_prop(2, self.Y, self.Ys, Bernoulli(), RBF, False)
#
#     def test_3_input_prop__bernoulli(self):
#         self.compare_input_prop(3, self.Y, self.Ys, Bernoulli(), RBF, False)
#
#
#     def compare_input_prop(self, L, Y, Ys, lik, Kern, analytic_exp):
#         m_svgp = SVGP(self.X, Y, Kern(self.D_X), lik, Z=self.X)
#         m_svgp.q_mu = self.q_mu
#         m_svgp.q_sqrt = self.q_sqrt
#
#         mean_svgp, var_svgp = m_svgp.predict_f(self.Xs)
#         test_lik_svgp = m_svgp.predict_density(self.Xs, Ys)
#
#         kerns = [Kern(self.D_X)]
#         for l in range(L-1):
#             kerns.append(Kern(2*self.D_X))
#
#         m_dgp = DGP(self.X, Y, self.X, kerns, lik,
#                     init_layers=init_layers_input_propagation,
#                     analytic_final_expectations=analytic_exp)
#
#         K = kerns[0].compute_K_symm(self.X)
#         L_inv_X = np.linalg.solve(np.linalg.cholesky(K), self.X)
#
#         for layer in m_dgp.layers[:-1]:
#             layer.kern.variance = 1e-24
#             layer.q_mu = L_inv_X
#             layer.q_sqrt = layer.q_sqrt.read_value() * 1e-12
#
#         m_dgp.layers[-1].q_mu = self.q_mu
#         m_dgp.layers[-1].q_sqrt = self.q_sqrt
#
#         mean_dgp, var_dgp = m_dgp.predict_f(self.Xs, 1)
#         test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)
#
#         assert_allclose(mean_svgp, mean_dgp[0], atol=1e-3, rtol=1e-3)
#         assert_allclose(var_svgp, var_dgp[0],atol=1e-3, rtol=1e-3)
#         assert_allclose(test_lik_svgp, test_lik_dgp, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

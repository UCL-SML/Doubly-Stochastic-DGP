import unittest
import numpy as np

from numpy.testing import assert_allclose

from gpflow import settings, session_manager
from gpflow.models.svgp import SVGP
from gpflow.kernels import RBF, Matern52
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
        self.q_sqrt = np.random.randn(N, N, D_Y)
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

        m_dgp = DGP(self.X, Y, self.X, [kern], lik, init_layers=init_method)
        m_dgp.layers[0].q_mu = self.q_mu
        m_dgp.layers[0].q_sqrt = self.q_sqrt

        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_y(self.Xs, 1)
        test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)

        assert_allclose(L_svgp, L_dgp)

        assert_allclose(mean_svgp, mean_dgp[0])
        assert_allclose(var_svgp, var_dgp[0])
        assert_allclose(test_lik_svgp, test_lik_dgp)


class TestTwoLayerVsSingleLayer(unittest.TestCase):
    def test_linear_mean_gaussian(self):
        Ns, N, D_X, D_Y = 2, 2, 1, 1

        X = np.random.uniform(low=-5, high=5, size=(N, D_X))
        Y = np.random.randn(N, D_Y)
        Xs = np.random.uniform(low=-5, high=5, size=(Ns, D_X))
        Ys = np.random.randn(Ns, D_Y)

        lik = Gaussian()
        Kern = RBF
        q_mu = np.random.randn(N, D_Y)
        q_sqrt = np.random.randn(N, N, D_Y)

        m_svgp = SVGP(X, Y, Kern(D_X), lik, Z=X)
        m_svgp.q_mu = q_mu
        m_svgp.q_sqrt = q_sqrt

        L_svgp = m_svgp.compute_log_likelihood()
        mean_svgp, var_svgp = m_svgp.predict_f(Xs)
        test_lik_svgp = m_svgp.predict_density(Xs, Ys)

        init_method = init_layers_linear_mean_functions
        kerns = [Kern(D_X), Kern(D_X)]

        m_dgp = DGP(X, Y, X, kerns, lik, init_layers=init_method)

        m_dgp.layers[0].kern.variance = 1e-12

        m_dgp.layers[-1].q_mu = q_mu
        m_dgp.layers[-1].q_sqrt = q_sqrt

        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_f(Xs, 1)
        test_lik_dgp = m_dgp.predict_density(Xs, Ys, 1)

        assert_allclose(L_svgp, L_dgp, atol=1e-3, rtol=1e-3)

        assert_allclose(mean_svgp, mean_dgp[0], atol=1e-3, rtol=1e-3)
        assert_allclose(var_svgp, var_dgp[0],atol=1e-3, rtol=1e-3)
        assert_allclose(test_lik_svgp, test_lik_dgp, atol=1e-3, rtol=1e-3)


    def test_input_prop_gaussian(self):
        Ns, N, D_X, D_Y = 2, 2, 1, 1

        X = np.random.uniform(low=-5, high=5, size=(N, D_X))
        Y = np.random.randn(N, D_Y)
        Xs = np.random.uniform(low=-5, high=5, size=(Ns, D_X))
        Ys = np.random.randn(Ns, D_Y)

        lik = Gaussian()
        Kern = RBF
        q_mu = np.random.randn(N, D_Y)
        q_sqrt = np.random.randn(N, N, D_Y)

        m_svgp = SVGP(X, Y, Kern(D_X), lik, Z=X)
        m_svgp.q_mu = q_mu
        m_svgp.q_sqrt = q_sqrt

        mean_svgp, var_svgp = m_svgp.predict_f(Xs)
        test_lik_svgp = m_svgp.predict_density(Xs, Ys)

        init_method = init_layers_input_propagation
        kerns = [Kern(D_X), Kern(2*D_X)]

        m_dgp = DGP(X, Y, X, kerns, lik, init_layers=init_method)

        m_dgp.layers[0].kern.variance = 1e-12

        K = kerns[0].compute_K_symm(X)
        m_dgp.layers[1].q_mu = np.linalg.solve(np.linalg.cholesky(K), X)
        m_dgp.layers[0].q_sqrt = m_dgp.layers[0].q_sqrt.read_value() * 1e-12

        m_dgp.layers[-1].q_mu = q_mu
        m_dgp.layers[-1].q_sqrt = q_sqrt

        mean_dgp, var_dgp = m_dgp.predict_f(Xs, 1)
        test_lik_dgp = m_dgp.predict_density(Xs, Ys, 1)

        assert_allclose(mean_svgp, mean_dgp[0], atol=1e-3, rtol=1e-3)
        assert_allclose(var_svgp, var_dgp[0],atol=1e-3, rtol=1e-3)
        assert_allclose(test_lik_svgp, test_lik_dgp, atol=1e-3, rtol=1e-3)


    def test_linear_mean_bernoulli(self):
        Ns, N, D_X, D_Y = 2, 2, 1, 1

        X = np.random.uniform(low=-5, high=5, size=(N, D_X))
        Y = np.random.choice([1., -1], N).reshape(N, 1)
        Xs = np.random.uniform(low=-5, high=5, size=(Ns, D_X))
        Ys = np.random.choice([1., -1], Ns).reshape(Ns, 1)

        lik = Bernoulli()
        Kern = RBF
        q_mu = np.random.randn(N, D_Y)
        q_sqrt = np.random.randn(N, N, D_Y)

        m_svgp = SVGP(X, Y, Kern(D_X), lik, Z=X)
        m_svgp.q_mu = q_mu
        m_svgp.q_sqrt = q_sqrt

        L_svgp = m_svgp.compute_log_likelihood()
        mean_svgp, var_svgp = m_svgp.predict_f(Xs)
        test_lik_svgp = m_svgp.predict_density(Xs, Ys)

        init_method = init_layers_linear_mean_functions
        kerns = [Kern(D_X), Kern(D_X)]

        m_dgp = DGP(X, Y, X, kerns, lik, init_layers=init_method)

        m_dgp.layers[0].kern.variance = 1e-12

        m_dgp.layers[-1].q_mu = q_mu
        m_dgp.layers[-1].q_sqrt = q_sqrt

        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_f(Xs, 1)
        test_lik_dgp = m_dgp.predict_density(Xs, Ys, 1)

        assert_allclose(L_svgp, L_dgp, atol=1e-3, rtol=1e-3)

        assert_allclose(mean_svgp, mean_dgp[0], atol=1e-3, rtol=1e-3)
        assert_allclose(var_svgp, var_dgp[0],atol=1e-3, rtol=1e-3)
        assert_allclose(test_lik_svgp, test_lik_dgp, atol=1e-3, rtol=1e-3)


    def test_input_prop_gaussian_likelihood(self):
        Ns, N, D_X, D_Y = 2, 2, 1, 1

        X = np.random.uniform(low=-5, high=5, size=(N, D_X))
        Y = np.random.choice([1., -1], N).reshape(N, 1)
        Xs = np.random.uniform(low=-5, high=5, size=(Ns, D_X))
        Ys = np.random.choice([1., -1], Ns).reshape(Ns, 1)

        lik = Bernoulli()
        Kern = RBF
        q_mu = np.random.randn(N, D_Y)
        q_sqrt = np.random.randn(N, N, D_Y)

        m_svgp = SVGP(X, Y, Kern(D_X), lik, Z=X)
        m_svgp.q_mu = q_mu
        m_svgp.q_sqrt = q_sqrt

        mean_svgp, var_svgp = m_svgp.predict_f(Xs)
        test_lik_svgp = m_svgp.predict_density(Xs, Ys)

        init_method = init_layers_input_propagation
        kerns = [Kern(D_X), Kern(2*D_X)]

        m_dgp = DGP(X, Y, X, kerns, lik, init_layers=init_method)

        m_dgp.layers[0].kern.variance = 1e-12

        K = kerns[0].compute_K_symm(X)
        m_dgp.layers[1].q_mu = np.linalg.solve(np.linalg.cholesky(K), X)
        m_dgp.layers[0].q_sqrt = m_dgp.layers[0].q_sqrt.read_value() * 1e-12

        m_dgp.layers[-1].q_mu = q_mu
        m_dgp.layers[-1].q_sqrt = q_sqrt

        mean_dgp, var_dgp = m_dgp.predict_f(Xs, 1)
        test_lik_dgp = m_dgp.predict_density(Xs, Ys, 1)

        assert_allclose(mean_svgp, mean_dgp[0], atol=1e-3, rtol=1e-3)
        assert_allclose(var_svgp, var_dgp[0],atol=1e-3, rtol=1e-3)
        assert_allclose(test_lik_svgp, test_lik_dgp, atol=1e-3, rtol=1e-3)




if __name__ == '__main__':
    unittest.main()

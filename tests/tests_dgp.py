import unittest
import numpy as np

from numpy.testing import assert_allclose

from gpflow import settings, session_manager
from gpflow.models.svgp import SVGP
from gpflow.kernels import Matern52
from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass

from doubly_stochastic_dgp.dgp import DGP

np.random.seed(0)


class TestVsSingleLayer(unittest.TestCase):
    def setUp(self):
        Ns, N, D_X, D_Y = 5, 4, 2, 3

        self.X = np.random.uniform(size=(N, D_X))
        self.Xs = np.random.uniform(size=(Ns, D_X))
        self.q_mu = np.random.randn(N, D_Y)
        self.q_sqrt = np.random.randn(D_Y, N, N)
        self.D_Y = D_Y

    def test_gaussian(self):
        lik = Gaussian()
        lik.variance = 0.01
        N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        Y = np.random.randn(N, D_Y)
        Ys = np.random.randn(Ns, D_Y)
        for L in [1, 2]:
            self.compare_to_single_layer(Y, Ys, lik, L)

    def test_bernoulli(self):
        lik = Bernoulli()
        N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        Y = np.random.choice([-1., 1.], N*D_Y).reshape(N, D_Y)
        Ys = np.random.choice([-1., 1.], Ns*D_Y).reshape(Ns, D_Y)
        for L in [1, 2]:
            self.compare_to_single_layer(Y, Ys, lik, L)

    def test_multiclass(self):
        K = 3
        lik = MultiClass(K)
        N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        Y = np.random.choice([0., 1., 2.], N * 1).reshape(N, 1)
        Ys = np.random.choice([0., 1., 2.], Ns * 1).reshape(Ns, 1)
        for L in [1, 2]:
            self.compare_to_single_layer(Y, Ys, lik, L, num_outputs=K)

    def compare_to_single_layer(self, Y, Ys, lik, L, num_outputs=None):
        kern = Matern52(self.X.shape[1], lengthscales=0.1)

        m_svgp = SVGP(self.X, Y, kern, lik, Z=self.X, num_latent=num_outputs)
        m_svgp.q_mu = self.q_mu
        m_svgp.q_sqrt = self.q_sqrt

        L_svgp = m_svgp.compute_log_likelihood()
        mean_svgp, var_svgp = m_svgp.predict_y(self.Xs)
        test_lik_svgp = m_svgp.predict_density(self.Xs, Ys)
        pred_m_svgp, pred_v_svgp = m_svgp.predict_f(self.Xs)
        pred_mfull_svgp, pred_vfull_svgp = m_svgp.predict_f_full_cov(self.Xs)

        kerns = []
        for _ in range(L-1):
            kerns.append(Matern52(self.X.shape[1], lengthscales=0.1, variance=2e-6))
        kerns.append(Matern52(self.X.shape[1], lengthscales=0.1))

        m_dgp = DGP(self.X, Y, self.X, kerns, lik, num_samples=2, num_outputs=num_outputs)
        m_dgp.layers[-1].q_mu = self.q_mu
        m_dgp.layers[-1].q_sqrt = self.q_sqrt

        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_y(self.Xs, 1)
        test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)

        pred_m_dgp, pred_v_dgp = m_dgp.predict_f(self.Xs, 1)
        pred_mfull_dgp, pred_vfull_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)

        if L == 1: # these should all be exactly the same
            atol = 1e-7
            rtol = 1e-7
        else:  # jitter makes these not exactly equal
            atol = 1e-1
            rtol = 1e-2

        assert_allclose(L_svgp, L_dgp, rtol=rtol, atol=atol)

        assert_allclose(mean_svgp, mean_dgp[0], rtol=rtol, atol=atol)
        assert_allclose(var_svgp, var_dgp[0], rtol=rtol, atol=atol)
        assert_allclose(test_lik_svgp, test_lik_dgp, rtol=rtol, atol=atol)

        assert_allclose(pred_m_dgp[0], pred_m_svgp, rtol=rtol, atol=atol)
        assert_allclose(pred_v_dgp[0], pred_v_svgp, rtol=rtol, atol=atol)
        assert_allclose(pred_mfull_dgp[0], pred_mfull_svgp, rtol=rtol, atol=atol)
        assert_allclose(pred_vfull_dgp[0], pred_vfull_svgp, rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()

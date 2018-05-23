import unittest
import numpy as np

from gpflow import settings as _settings
from gpflow import session_manager as _session_manager
from gpflow.training import NatGradOptimizer

custom_config = _settings.get_settings()
custom_config.numerics.jitter_level = 1e-6

with _settings.temp_settings(custom_config),\
     _session_manager.get_session().as_default():

    from numpy.testing import assert_allclose

    from gpflow.models.svgp import SVGP
    from gpflow.kernels import Matern52, RBF
    from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass
    from gpflow.training import ScipyOptimizer

    from doubly_stochastic_dgp.dgp import DGP, DGP_Base, DGP_Quad
    from doubly_stochastic_dgp.layer_initializations import init_layers_linear
    np.random.seed(0)

    class TestVsSingleLayer(unittest.TestCase):
        def setUp(self):
            Ns, N, D_X, D_Y = 20, 19, 2, 3
            np.random.seed(0)
            self.X = np.random.uniform(size=(N, D_X))
            self.Xs = np.random.uniform(size=(Ns, D_X))
            self.q_mu = np.random.randn(N, D_Y)
            self.q_sqrt = 0.001*np.eye(N)[None, :, :] * np.ones((D_Y, 1, 1))#np.tril(np.random.randn(D_Y, N, N))**2

            self.D_Y = D_Y

        def test_gaussian(self):
            lik = Gaussian()
            lik.variance = 0.01
            N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
            Y = np.random.randn(N, D_Y)
            Ys = np.random.randn(Ns, D_Y)
            for L in [1, ]:
                for white in [True]:
                    self.compare_to_single_layer(Y, Ys, lik, L, white)

        # def test_bernoulli(self):
        #     lik = Bernoulli()
        #     N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        #     Y = np.random.choice([-1., 1.], N*D_Y).reshape(N, D_Y)
        #     Ys = np.random.choice([-1., 1.], Ns*D_Y).reshape(Ns, D_Y)
        #     for L in [1, 2]:
        #         self.compare_to_single_layer(Y, Ys, lik, L)
        #
        # def test_multiclass(self):
        #     K = 3
        #     lik = MultiClass(K)
        #     N, Ns, D_Y = self.X.shape[0], self.Xs.shape[0], self.D_Y
        #     Y = np.random.choice([0., 1., 2.], N * 1).reshape(N, 1)
        #     Ys = np.random.choice([0., 1., 2.], Ns * 1).reshape(Ns, 1)
        #     for L in [1, 2]:
        #         self.compare_to_single_layer(Y, Ys, lik, L, num_outputs=K)

        def compare_to_single_layer(self, Y, Ys, lik, L, white, num_outputs=None):
            kern = Matern52(self.X.shape[1], lengthscales=0.5)

            m_svgp = SVGP(self.X, Y, kern, lik, Z=self.X, whiten=white, num_latent=num_outputs)
            m_svgp.q_mu = self.q_mu
            m_svgp.q_sqrt = self.q_sqrt

            #
            # L_svgp = m_svgp.compute_log_likelihood()
            # mean_svgp, var_svgp = m_svgp.predict_y(self.Xs)
            # test_lik_svgp = m_svgp.predict_density(self.Xs, Ys)
            # pred_m_svgp, pred_v_svgp = m_svgp.predict_f(self.Xs)
            # pred_mfull_svgp, pred_vfull_svgp = m_svgp.predict_f_full_cov(self.Xs)
            #
            # kerns = []
            # for _ in range(L-1):
            #     kerns.append(Matern52(self.X.shape[1], lengthscales=0.5, variance=2e-6))
            # kerns.append(kern)
            #
            # m_dgp = DGP(self.X, Y, self.X, kerns, lik, white=white, num_samples=2, num_outputs=num_outputs)
            # m_dgp.layers[-1].q_mu = self.q_mu
            # m_dgp.layers[-1].q_sqrt = self.q_sqrt
            #
            # L_dgp = m_dgp.compute_log_likelihood()
            # mean_dgp, var_dgp = m_dgp.predict_y(self.Xs, 1)
            # test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)
            #
            # pred_m_dgp, pred_v_dgp = m_dgp.predict_f(self.Xs, 1)
            # pred_mfull_dgp, pred_vfull_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)
            #
            # if L == 1: # these should all be exactly the same
            #     atol = 1e-7
            #     rtol = 1e-7
            # else:  # jitter makes these not exactly equal
            #     atol = 1e-1
            #     rtol = 1e-2
            #
            # # assert_allclose(L_svgp, L_dgp, rtol=rtol, atol=atol)
            # #
            # # assert_allclose(mean_svgp, mean_dgp[0], rtol=rtol, atol=atol)
            # # assert_allclose(var_svgp, var_dgp[0], rtol=rtol, atol=atol)
            # # assert_allclose(test_lik_svgp, test_lik_dgp, rtol=rtol, atol=atol)
            # #
            # # assert_allclose(pred_m_dgp[0], pred_m_svgp, rtol=rtol, atol=atol)
            # # assert_allclose(pred_v_dgp[0], pred_v_svgp, rtol=rtol, atol=atol)
            # # assert_allclose(pred_mfull_dgp[0], pred_mfull_svgp, rtol=rtol, atol=atol)
            # # assert_allclose(pred_vfull_dgp[0], pred_vfull_svgp, rtol=rtol, atol=atol)
            print(m_svgp.compute_log_likelihood())

            NatGradOptimizer(gamma=1.).minimize(m_svgp, var_list=[[m_svgp.q_mu, m_svgp.q_sqrt]], maxiter=1)
            print(m_svgp.compute_log_likelihood())


    # class TestQuad(unittest.TestCase):
    #     def test_quadrature(self):
    #         N = 50
    #         X = np.random.uniform(size=(N, 1))
    #         Y = np.sin(20*X) + np.random.randn(*X.shape) * 0.001
    #
    #         kernels = lambda : [RBF(1, lengthscales=0.1), RBF(1, lengthscales=0.1)]
    #         layers = lambda : init_layers_linear(X, Y, X, kernels())
    #         def lik():
    #             l = Gaussian()
    #             l.variance = 0.01
    #             return l
    #
    #         m_stochastic = DGP_Base(X, Y, lik(), layers(), num_samples=10000)
    #         m_quad = DGP_Quad(X, Y, lik(), layers(), H=100)
    #
    #         # q_mu_0 = np.random.randn(N, 1)
    #         # q_sqrt_0 = np.random.randn(1, N, N)
    #         #
    #         # q_mu_1 = np.random.randn(N, 1)
    #         # q_sqrt_1 = np.random.randn(1, N, N)
    #
    #         for model in m_quad, m_stochastic:
    #             model.set_trainable(False)
    #             for layer in model.layers:
    #                 layer.q_mu.set_trainable(True)
    #                 layer.q_sqrt.set_trainable(True)
    #
    #         ScipyOptimizer().minimize(m_quad, maxiter=100)
    #
    #         q_mu_0 = m_quad.layers[0].q_mu.read_value()
    #         q_sqrt_0 = m_quad.layers[0].q_sqrt.read_value()
    #
    #         q_mu_1 = m_quad.layers[1].q_mu.read_value()
    #         q_sqrt_1 = m_quad.layers[1].q_sqrt.read_value()
    #
    #
    #         for model in m_stochastic,:#, m_quad:
    #             model.layers[0].q_mu = q_mu_0
    #             model.layers[0].q_sqrt = q_sqrt_0
    #
    #             model.layers[1].q_mu = q_mu_1
    #             model.layers[1].q_sqrt = q_sqrt_1
    #
    #         Ls_quad = [m_quad.compute_log_likelihood() for _ in range(2)]
    #         Ls_stochastic = [m_stochastic.compute_log_likelihood() for _ in range(100)]
    #
    #         assert_allclose(Ls_quad[0], Ls_quad[1])
    #         m = np.average(Ls_stochastic)
    #         std_err = np.std(Ls_stochastic)/(float(len(Ls_stochastic))**0.5)
    #         print(m)
    #         print(std_err)
    #         print(Ls_quad[0])
    #         assert_allclose(Ls_quad[0], m)


    if __name__ == '__main__':
        unittest.main()

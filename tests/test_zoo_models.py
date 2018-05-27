import unittest
import numpy as np

from numpy.testing import assert_allclose

from doubly_stochastic_dgp.layers import GPMC_Layer, GPR_Layer

from gpflow import settings as _settings
from gpflow import session_manager as _session_manager

custom_config = _settings.get_settings()
custom_config.numerics.jitter_level = 1e-12

with _settings.temp_settings(custom_config),\
     _session_manager.get_session().as_default():


    from gpflow.models import SVGP, GPR
    from gpflow.kernels import Matern52, RBF
    from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass
    from gpflow.training import ScipyOptimizer
    from gpflow.mean_functions import Zero, Identity, Linear, Constant
    from gpflow.training import NatGradOptimizer

    from gpflow import settings

    from doubly_stochastic_dgp.model_zoo import DGP_Heinonen

    from doubly_stochastic_dgp.dgp import DGP, DGP_Base, DGP_Quad
    from doubly_stochastic_dgp.layer_initializations import init_layers_linear
    np.random.seed(0)


    class TestHeinonen(unittest.TestCase):
        def setUp(self):
            Ns, N, D_X, D_Y = 5, 6, 3, 2

            self.X = np.random.uniform(size=(N, D_X))
            self.Xs = self.X #np.random.uniform(size=(Ns, D_X))

            self.D_Y = D_Y

        def test_vs_single_layer(self):
            lik = Gaussian()
            lik_var = 0.01
            lik.variance = lik_var
            N, Ns, D_Y, D_X = self.X.shape[0], self.Xs.shape[0], self.D_Y, self.X.shape[1]
            Y = np.random.randn(N, D_Y)
            Ys = np.random.randn(Ns, D_Y)

            kern = Matern52(self.X.shape[1], lengthscales=0.5)
            # mf = Linear(A=np.random.randn(D_X, D_Y), b=np.random.randn(D_Y))
            mf = Zero()
            m_gpr = GPR(self.X, Y, kern, mean_function=mf)
            m_gpr.likelihood.variance = lik_var
            mean_gpr, var_gpr = m_gpr.predict_y(self.Xs)
            test_lik_gpr = m_gpr.predict_density(self.Xs, Ys)
            pred_m_gpr, pred_v_gpr = m_gpr.predict_f(self.Xs)
            pred_mfull_gpr, pred_vfull_gpr = m_gpr.predict_f_full_cov(self.Xs)

            kerns = []
            kerns.append(Matern52(self.X.shape[1], lengthscales=0.5, variance=1e-1))
            kerns.append(kern)

            layer0 = GPMC_Layer(kerns[0], self.X.copy(), D_X, Identity())
            layer1 = GPR_Layer(kerns[1], mf, D_Y)
            m_dgp = DGP_Heinonen(self.X, Y, lik, [layer0, layer1])


            mean_dgp, var_dgp = m_dgp.predict_y(self.Xs, 1)
            test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)
            pred_m_dgp, pred_v_dgp = m_dgp.predict_f(self.Xs, 1)
            pred_mfull_dgp, pred_vfull_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)

            tol = 1e-4
            assert_allclose(mean_dgp[0], mean_gpr, atol=tol, rtol=tol)
            assert_allclose(test_lik_dgp, test_lik_gpr, atol=tol, rtol=tol)
            assert_allclose(pred_m_dgp[0], pred_m_gpr, atol=tol, rtol=tol)
            assert_allclose(pred_mfull_dgp[0], pred_mfull_gpr, atol=tol, rtol=tol)
            assert_allclose(pred_vfull_dgp[0], pred_vfull_gpr, atol=tol, rtol=tol)

        def test_vs_DGP2(self):
            lik = Gaussian()
            lik_var = 0.1
            lik.variance = lik_var
            N, Ns, D_Y, D_X = self.X.shape[0], self.Xs.shape[0], self.D_Y, self.X.shape[1]

            q_mu = np.random.randn(N, D_X)

            Y = np.random.randn(N, D_Y)
            Ys = np.random.randn(Ns, D_Y)

            kern1 = Matern52(self.X.shape[1], lengthscales=0.5)
            kern2 = Matern52(self.X.shape[1], lengthscales=0.5)
            kerns = [kern1, kern2]
            # mf = Linear(A=np.random.randn(D_X, D_Y), b=np.random.randn(D_Y))

            mf = Zero()
            m_dgp = DGP(self.X, Y, self.X, kerns, lik, mean_function=mf, white=True)
            m_dgp.layers[0].q_mu = q_mu
            m_dgp.layers[0].q_sqrt = m_dgp.layers[0].q_sqrt.read_value() * 1e-24

            Fs, ms, vs = m_dgp.predict_all_layers(self.Xs, 1)
            Z = self.X.copy()
            Z[:len(self.Xs)] = ms[0][0]
            m_dgp.layers[1].feature.Z = Z  # need to put the inducing points in the right place

            var_list = [[m_dgp.layers[1].q_mu, m_dgp.layers[1].q_sqrt]]
            NatGradOptimizer(gamma=1).minimize(m_dgp, var_list=var_list, maxiter=1)

            mean_dgp, var_dgp = m_dgp.predict_y(self.Xs, 1)
            test_lik_dgp = m_dgp.predict_density(self.Xs, Ys, 1)
            pred_m_dgp, pred_v_gpr = m_dgp.predict_f(self.Xs, 1)
            pred_mfull_dgp, pred_vfull_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)

            # mean_functions = [Identity(), mf]
            layer0 = GPMC_Layer(kerns[0], self.X.copy(), D_X, Identity())
            layer1 = GPR_Layer(kerns[1], mf, D_Y)

            m_heinonen = DGP_Heinonen(self.X, Y, lik, [layer0, layer1])

            m_heinonen.layers[0].q_mu = q_mu

            mean_heinonen, var_heinonen = m_heinonen.predict_y(self.Xs, 1)
            test_lik_heinonen = m_heinonen.predict_density(self.Xs, Ys, 1)
            pred_m_heinonen, pred_v_heinonen = m_heinonen.predict_f(self.Xs, 1)
            pred_mfull_heinonen, pred_vfull_heinonen = m_heinonen.predict_f_full_cov(self.Xs, 1)

            tol = 1e-4
            assert_allclose(mean_dgp, mean_heinonen, atol=tol, rtol=tol)
            assert_allclose(test_lik_dgp, test_lik_heinonen, atol=tol, rtol=tol)
            assert_allclose(pred_m_dgp, pred_m_heinonen, atol=tol, rtol=tol)
            assert_allclose(pred_mfull_dgp, pred_mfull_heinonen, atol=tol, rtol=tol)
            assert_allclose(pred_vfull_dgp, pred_vfull_heinonen, atol=tol, rtol=tol)





    if __name__ == '__main__':
        unittest.main()

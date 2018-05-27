import unittest
import numpy as np

from numpy.testing import assert_allclose

from gpflow.models.svgp import SVGP
from gpflow.models import GPR
from gpflow.kernels import Matern52, RBF
from gpflow.likelihoods import Gaussian, Bernoulli, MultiClass
from gpflow.training import ScipyOptimizer
from gpflow.training import NatGradOptimizer
from doubly_stochastic_dgp.dgp import DGP, DGP_Base, DGP_Quad
from doubly_stochastic_dgp.model_zoo import DGP_Collapsed
from doubly_stochastic_dgp.layers import SGPR_Layer
from doubly_stochastic_dgp.layer_initializations import init_layers_linear

np.random.seed(100)

class TestVsSingleLayer(unittest.TestCase):
    def setUp(self):
        Ns, N, M, D_X, D_Y = 5, 4, 2, 3, 2
        self.lik_var = 0.1

        self.X = np.random.uniform(size=(N, D_X))
        self.Y = np.random.uniform(size=(N, D_Y))
        self.Z = np.random.uniform(size=(M, D_Y))
        self.Xs = np.random.uniform(size=(Ns, D_X))
        self.D_Y = D_Y

    def test_single_layer(self):
        kern = RBF(1, lengthscales=0.1)
        layers = init_layers_linear(self.X, self.Y, self.X, [kern])

        lik = Gaussian()
        lik.variance = self.lik_var

        last_layer = SGPR_Layer(layers[-1].kern,
                                layers[-1].feature.Z.read_value(),
                                self.D_Y,
                                layers[-1].mean_function)
        layers = layers[:-1] + [last_layer]

        m_dgp = DGP_Collapsed(self.X, self.Y, lik, layers)
        L_dgp = m_dgp.compute_log_likelihood()
        mean_dgp, var_dgp = m_dgp.predict_f_full_cov(self.Xs, 1)

        m_exact = GPR(self.X, self.Y, kern)
        m_exact.likelihood.variance = self.lik_var
        L_exact = m_exact.compute_log_likelihood()
        mean_exact, var_exact = m_exact.predict_f_full_cov(self.Xs)

        assert_allclose(L_dgp, L_exact, atol=1e-5, rtol=1e-5)
        assert_allclose(mean_dgp[0], mean_exact, atol=1e-5, rtol=1e-5)
        assert_allclose(var_dgp[0], var_exact, atol=1e-5, rtol=1e-5)


class TestVsNatGrads(unittest.TestCase):
    def test_2layer_vs_nat_grad(self):
        Ns, N, M = 5, 1, 50
        D_X, D_Y = 1, 1

        lik_var = 0.1

        X = np.random.uniform(size=(N, D_X))
        Y = np.random.uniform(size=(N, D_Y))
        Z = np.random.uniform(size=(M, D_Y))
        Xs = np.random.uniform(size=(Ns, D_X))

        Z[:N, :] = X[:M, :]

        def kerns():
            return [RBF(D_X, lengthscales=0.1),
                    RBF(D_X, lengthscales=0.5)]
        layers_col = init_layers_linear(X, Y, Z, kerns())
        layers_ng = init_layers_linear(X, Y, Z, kerns())

        def lik():
            l = Gaussian()
            l.variance = lik_var
            return l

        last_layer = SGPR_Layer(layers_col[-1].kern,
                                layers_col[-1].feature.Z.read_value(),
                                D_Y,
                                layers_col[-1].mean_function)

        layers_col = layers_col[:-1] + [last_layer]
        m_col = DGP_Collapsed(X, Y, lik(), layers_col)
        m_ng = DGP_Quad(X, Y, lik(), layers_ng, H=200)

        q_mu1 = np.random.randn(M, D_X)
        q_sqrt1 = np.random.randn(M, M)
        q_sqrt1 = np.tril(q_sqrt1)[None, :, :]

        for m in m_col, m_ng:
            m.layers[0].q_mu = q_mu1
            m.layers[0].q_sqrt = q_sqrt1

        p = [[m_ng.layers[-1].q_mu, m_ng.layers[-1].q_sqrt]]
        NatGradOptimizer(gamma=1.).minimize(m_ng, var_list=p, maxiter=1)


        assert_allclose(m_col.compute_log_likelihood(),
                        m_ng.compute_log_likelihood())


if __name__ == '__main__':
    unittest.main()

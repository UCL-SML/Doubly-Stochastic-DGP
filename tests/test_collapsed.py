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

        print('m col')
        for _ in range(1):
            print(m_col.compute_log_likelihood())

        print('m ng')
        for _ in range(3):
            print(m_ng.compute_log_likelihood())

        assert_allclose(m_col.compute_log_likelihood(),
                        m_ng.compute_log_likelihood())
        # print('diff {}'.format(m_ng.compute_log_likelihood() - m_col.compute_log_likelihood()))

        # print(m_ng)
        # print(m_col)

if __name__ == '__main__':
    unittest.main()

#
# import sys
# sys.path.append('../../')
# sys.path.append('/homes/hrs13/Documents/github/DNSGP/experiments/posterior_validation')
# sys.path.append('/homes/hrs13/Documents/github/DNSGP/')
#
# from doubly_stochastic_dgp.dgp import DGP
# import numpy as np
# import tensorflow as tf
# # import matplotlib.pyplot as plt
#
# import time
#
# from natural_gradients_gpflow.natural_gradient_optimizers import NaturalGradientsWithAdamOptimizer as NGO
#
# from models_to_compare import create_model
# from doubly_stochastic_dgp.dgp import DGP_Quad, DGP_Base, DGP_Collapsed, DGP_Quad
# from doubly_stochastic_dgp.layers import SVGP_Layer, SGPMC_Layer
#
# from gpflow.training import HMC
#
# from experiment_args import FLAGS
#
#
# class Flags:
#     def __init__(self):
#         self.flags = {}
#
#     def DEFINE_integer(self, a, b, c):
#         self.flags.update({a: int(b)})
#
#     def DEFINE_string(self, a, b, c):
#         self.flags.update({a: str(b)})
#
#     def DEFINE_float(self, a, b, c):
#         self.flags.update({a: float(b)})
#
#
# flags = Flags()
#
# flags.DEFINE_integer('Ns', 100, 'number of test points')
#
# flags.DEFINE_integer('N', 50, 'number of training points')
# flags.DEFINE_integer('M', 11, 'number of inducing points')
# flags.DEFINE_integer('L', 2, 'number of layers')
#
# flags.DEFINE_integer('H', 10, 'number of samples/quad points')
# flags.DEFINE_string('model_type', 'DLGP', 'model type: DGP, DLGP')
# flags.DEFINE_string('inference_type', 'HMC', 'type of inference: SVI, VI, HMC')
#
# flags.DEFINE_integer('num_samples', 1, 'number of samples for prediction')
#
# flags.DEFINE_float('inner_layer_mag', 0.5, 'magnitude of inner layer')
#
# flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
# flags.DEFINE_integer('iterations', 1, 'iterations')
#
# flags.DEFINE_integer('hmc_samples', 1, 'iterations')
# flags.DEFINE_integer('hmc_pre_train_iterations', 1, 'iterations')
# flags.DEFINE_integer('hmc_burn_iterations', 1, 'burn-in iterations')
# flags.DEFINE_float('hmc_eps', 0.01, 'HMC epsilon size')
# flags.DEFINE_integer('hmc_thin', 1, 'thining factor for HMC')
#
#
# class Bunch(object):
#     def __init__(self, adict):
#         self.__dict__.update(adict)
#
# FLAGS = Bunch(flags.flags)
#
# np.random.seed(0)
#
# X = np.random.uniform(-1, 1, FLAGS.N).reshape(-1, 1)
# Xs = np.linspace(-1, 1, FLAGS.Ns).reshape(-1, 1)
# Z = np.linspace(-1, 1, FLAGS.M).reshape(-1, 1)
# np.random.shuffle(Z)
#
# def make_model(model_type, Model, Layer, **kw):
#     return create_model(model_type, Model, X, Xs, Z,
#                         FLAGS.L, Layer, FLAGS.inner_layer_mag, **kw)
#
# def init_models(model_type):
#     model_SVI, Fs_real, X_Xs = make_model(model_type, DGP_Base, SVGP_Layer,
#                                           num_samples=1,
#                                           full=True)
#     model_HMC, _ = make_model(model_type, DGP_Collapsed, SGPMC_Layer, H=FLAGS.H)
#
#     model_HMC.Y = model_SVI.Y.read_value()
#     return model_SVI, model_HMC, X_Xs
#
#
# # model_dgp_SVI, model_dgp_HMC, X_dgp_Xs = init_models('DGP')
#
# model_dlgp_SVI, model_dlgp_HMC, X_dlgp_Xs = init_models('DLGP')
# Fs, ms, vs = model_dlgp_HMC.predict_all_layers(Xs, 1)
# print(ms[0].shape)
#
#

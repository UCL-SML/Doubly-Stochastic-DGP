# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:46:18 2017

@author: hrs13
"""


import unittest
import numpy as np


import sys
sys.path.append('../src')
sys.path.append('../../DNSGP/GPflow')

from GPflow.svgp import SVGP
from GPflow.likelihoods import Gaussian
from GPflow.kernels import RBF

from GPflow.param import Param
from GPflow.transforms import Log1pe
from GPflow.gplvm import BayesianGPLVM
from dgp import DGP
    
    
class DGPTests(unittest.TestCase):
    def test_vs_single_layer_GP(self):
        N, D_X, D_Y, M, Ns = 5, 4, 3, 5, 5
        
        X = np.random.randn(N, D_X)
        Y = np.random.randn(N, D_Y)
        Z = np.random.randn(M, D_X)
        
        Xs = np.random.randn(Ns, D_X)
        
        noise = np.random.gamma([1, ])
        ls = np.random.gamma([D_X, ])
        s = np.random.gamma([D_Y, ])
        q_mu = np.random.randn(M, D_Y)
        q_sqrt = np.random.randn(M, M, D_Y)
        
        m_svgp = SVGP(X, Y, RBF(D_X), Gaussian(), Z,
                      q_diag=False, whiten=True)
        
        m_svgp.kern.lengthscales = ls
        m_svgp.kern.variance = s
        m_svgp.likelihood.variance = noise
        m_svgp.q_mu = q_mu
        m_svgp.q_sqrt = q_sqrt        

        def make_dgp_as_sgp(kernels):
            m_dgp = DGP(X, Y, Z, kernels, Gaussian())
            
            #set final layer to sgp
            m_dgp.layers[-1].kern.lengthscales = ls
            m_dgp.layers[-1].kern.variance = s
            m_dgp.likelihood.variance = noise
            m_dgp.layers[-1].q_mu = q_mu
            m_dgp.layers[-1].q_sqrt = q_sqrt
            
            # set other layers to identity 
            for layer in m_dgp.layers[:-1]:
#                1e-6 gives errors of 1e-3, so need to set right down
                layer.kern.variance.transform._lower = 1e-18
                layer.kern.variance = 1e-18
                
            return m_dgp
            
        m_dgp_1 = make_dgp_as_sgp([RBF(D_X), ])
        m_dgp_2 = make_dgp_as_sgp([RBF(D_X), RBF(D_X)])
        m_dgp_3 = make_dgp_as_sgp([RBF(D_X), RBF(D_X), RBF(D_X)])
        
        preds_svgp = m_svgp.predict_f(Xs)
        preds_dgp_1 = [mv[0] for mv in m_dgp_1.predict_f(Xs, 1)] 
        preds_dgp_2 = [mv[0] for mv in m_dgp_2.predict_f(Xs, 1)] 
        preds_dgp_3 = [mv[0] for mv in m_dgp_3.predict_f(Xs, 1)] 
        
        assert np.allclose(preds_svgp, preds_dgp_1)
        assert np.allclose(preds_svgp, preds_dgp_2)
        assert np.allclose(preds_svgp, preds_dgp_3)


#    def test_vs_gplvm(self):
#        N, D_X, D_Y, M, Ns = 5, 4, 3, 5, 5
#        
#        X_mean = np.random.randn(N, D_X)
#        X_var = np.random.gamma([N, D_X])
#        
#        Y = np.random.randn(N, D_Y)
##        Z = np.random.randn(M, D_X)
#        
#        Xs = np.random.randn(Ns, D_X)
#        
##        noise = np.random.gamma([1, ])
##        ls = np.random.gamma([D_X, ])
##        s = np.random.gamma([D_Y, ])
##        q_mu = np.random.randn(M, D_Y)
##        q_sqrt = np.random.randn(M, M, D_Y)
#        
#        
#        m_gplvm = BayesianGPLVM(X_mean, X_var, Y, RBF(D_X), M, 
#                                Z=None)
#        
#        m_gplvm.p
#
#
















































if __name__ == '__main__':
    unittest.main()



# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:29:13 2017

@author: hrs13
"""

import sys
sys.path.append('../src')
sys.path.append('../../DNSGP/GPflow')


import numpy as np
from GPflow.kernels import RBF
from GPflow.sgpr import SGPR

X = np.array(((0., 1.))).reshape([2, 1])

model = SGPR(X, X.copy(), RBF(1), X.copy())
for _ in range(10):
    print model.compute_log_likelihood()
    
import tensorflow as tf
print tf.__version__
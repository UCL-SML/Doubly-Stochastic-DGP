import sys
sys.path.append('../../DNSGP/GPflow/')


import numpy as np
from GPflow.likelihoods import Gaussian
from GPflow.kernels import RBF
from GPflow.sgpr import SGPR

X = np.array(((0., 1., 2.))).reshape([3, 1])

model = SGPR(X, X.copy(), RBF(1), X.copy())
for _ in range(10):
    print model.compute_log_likelihood()

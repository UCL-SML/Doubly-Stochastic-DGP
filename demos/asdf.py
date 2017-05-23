

import sys
sys.path.append('../src')
sys.path.append('../../DNSGP/GPflow/')

import numpy as np
import tensorflow as tf

from GPflow.likelihoods import MultiClass
from GPflow.kernels import RBF 
from GPflow.svgp import SVGP
from scipy.cluster.vq import kmeans2
from get_data import get_mnist_data

X, Y, Xs, Ys = get_mnist_data()

M = 100
Z = kmeans2(X, M, minit='points')[0]

m_sgp = SVPG(X, Y, RBF(784, lengthscales=2, variance=2), 
             MultiClass(10), Z, 
             num_latent=10, minibatch_size=100)

m_sgp.optimize(tf.train.AdamOptimizer(0.01), maxiter=1)
preds = np.argmax(m_sgp.predict_y(Xs)[0], 1)
print np.average(np.array(pred==Ys, dtype=float))





























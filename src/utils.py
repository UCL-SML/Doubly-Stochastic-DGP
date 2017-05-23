# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:36:25 2017

@author: hrs13
"""

import tensorflow as tf

from GPflow._settings import settings
jitter = settings.numerics.jitter_level
float_type = settings.dtypes.float_type

def tile_over_samples(X, S): 
    ones =  tf.ones([S, ] + [1, ] *  X.get_shape().ndims, dtype=X.dtype)
    return tf.expand_dims(X, 0) * ones

def shape_as_list(X):
    s = tf.shape(X)
    return [s[i] for i in range(X.get_shape().ndims)]

def normal_sample(mean, var, full_cov=False):
    if full_cov is False:
        z = tf.random_normal(tf.shape(mean), dtype=float_type)
        return mean + z * var ** 0.5
    else:
        S, N, D = shape_as_list(mean) # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
#        I = jitter * tf.eye(N, dtype=float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var)# + I)  # SDNN should be ok without as var already has jitter
        z = tf.random_normal([S, D, N, 1], dtype=float_type)
        f = mean + tf.matmul(chol, z)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND  

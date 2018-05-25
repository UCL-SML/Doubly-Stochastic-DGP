
import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvhermgauss
from gpflow import settings
float_type = settings.float_type

from doubly_stochastic_dgp.layers import SVGP_Layer


def init_layers_linear(X, Y, Z, kernels,
                       num_outputs=None,
                       mean_function=Zero(),
                       Layer=SVGP_Layer,
                       white=False):
    num_outputs = num_outputs or Y.shape[1]

    layers = []

    X_running, Z_running = X.copy(), Z.copy()
    for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):
        dim_in = kern_in.input_dim
        dim_out = kern_out.input_dim

        if dim_in == dim_out:
            mf = Identity()

        else:  # stepping down, use the pca projection
            _, _, V = np.linalg.svd(X_running, full_matrices=False)
            W = V[:dim_out, :].T

            mf = Linear(W)
            mf.set_trainable(False)

        layers.append(Layer(kern_in, Z_running, dim_out, mf, white=white))

        if dim_in != dim_out:
            Z_running = Z_running.dot(W)
            X_running = X_running.dot(W)

    # final layer
    layers.append(Layer(kernels[-1], Z_running, num_outputs, mean_function, white=white))
    return layers


def init_layers_input_prop(X, Y, Z, kernels,
                           num_outputs=None,
                           mean_function=Zero(),
                           Layer=SVGP_Layer,
                           white=False):
    num_outputs = num_outputs or Y.shape[1]
    D = X.shape[1]
    M = Z.shape[0]

    layers = []

    # class LayerInputProp(Layer):
    #     def __init__(self,  input_prop_dim, *args, **kw):
    #         Layer.__init__(self, *args, **kw)
    #         self.input_prop_dim = input_prop_dim
    #
    #     def sample_from_conditional(self, X, **kwargs):
    #         samples, mean, var = Layer.sample_from_conditional(self, X, **kwargs)
    #         shape = [tf.shape(X)[0], tf.shape(X)[1], self.input_prop_dim]
    #         X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)
    #
    #         samples = tf.concat([X_prop, samples], 2)
    #         mean = tf.concat([X_prop, mean], 2)
    #
    #         if kwargs['full_cov']:
    #             shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[1], tf.shape(var)[3])
    #             zeros = tf.ones(shape, dtype=settings.float_type) * 1e-6
    #             var = tf.concat([zeros, var], 3)
    #         else:
    #             var = tf.concat([tf.ones_like(X_prop) * 1e-6, var], 2)
    #         return samples, mean, var

    for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):
        dim_in = kern_in.input_dim
        dim_out = kern_out.input_dim - D
        std_in = kern_in.variance.read_value()**0.5
        pad = np.random.randn(M, dim_in - D) * 2. * std_in
        Z_padded = np.concatenate([Z, pad], 1)
        # layers.append(LayerInputProp(D, kern_in, Z_padded, dim_out, Zero(), white=white))
        layers.append(Layer(kern_in, Z_padded, dim_out, Zero(), white=white, input_prop_dim=D))

    dim_in = kernels[-1].input_dim
    std_in = kernels[-2].variance.read_value()**0.5 if dim_in > D else 1.
    pad = np.random.randn(M, dim_in - D) * 2. * std_in
    Z_padded = np.concatenate([Z, pad], 1)
    layers.append(Layer(kernels[-1], Z_padded, num_outputs, mean_function, white=white))
    return layers

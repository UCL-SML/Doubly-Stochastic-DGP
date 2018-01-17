# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from gpflow.params import ParamList
from gpflow.mean_functions import Linear, Zero

from doubly_stochastic_dgp.layer import Layer

def init_layers_linear_mean_functions(X, Y, Z, kernels, D_Y):
    M = Z.shape[0]
    dims = [k.input_dim for k in kernels] + [D_Y, ]

    X_running, Z_running = X.copy(), Z.copy()

    layers = []

    for dim_in, dim_out, kern in zip(dims[:-1], dims[1:], kernels):
        if dim_in == dim_out: # identity for same dims
            W = np.eye(dim_in)
        elif dim_in > dim_out: # use PCA mf for stepping down
            _, _, V = np.linalg.svd(X_running, full_matrices=False)
            W = V[:dim_out, :].T
        elif dim_in < dim_out: # identity + pad with zeros for stepping up
            I = np.eye(dim_in)
            zeros = np.zeros((dim_in, dim_out - dim_in))
            W = np.concatenate([I, zeros], 1)

        mean_function = Linear(A=W)
        mean_function.set_trainable(False)

        q_mu = np.zeros((M, dim_out))
        q_sqrt = np.eye(M)[None, :, :] * np.ones((dim_out, 1, 1))

        layers.append(Layer(kern, q_mu, q_sqrt, Z_running, mean_function))

        Z_running = Z_running.dot(W)
        X_running = X_running.dot(W)

    return ParamList(layers)


def init_layers_input_propagation(X, Y, Z, kernels, D_Y):
    D = X.shape[1]
    M = Z.shape[0]
    dims = [k.input_dim for k in kernels] + [D_Y, ]
    layers = []

    for l, (dim_in, dim_out, kern) in enumerate(zip(dims[:-1], dims[1:], kernels)):
        if l == 0:  # first layer, not input prop so Z is the original Z
            Z_layer = Z.copy()
            forward_prop = False

        else:  # other layers, need to input prop and append zeros to Z
            Z_layer = np.concatenate([Z.copy(), np.zeros((M, dim_in - D))], 1)
            forward_prop = True

        if l == len(dims) - 2:  # final layer, dim_out is Y_dim
            gp_output_dim = dim_out

        else:  # inner layers, dim_out = X_dim + gp_dim
            gp_output_dim = dim_out - D

        q_mu = np.zeros((M, gp_output_dim))
        q_sqrt = np.eye(M)[None, :, :] * np.ones((gp_output_dim, 1, 1))

        layers.append(Layer(kern, q_mu, q_sqrt, Z_layer, Zero(),
                            forward_propagate_inputs=forward_prop))

    return ParamList(layers)
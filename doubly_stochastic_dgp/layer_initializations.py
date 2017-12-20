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
from gpflow.mean_functions import Linear

from doubly_stochastic_dgp.layer import Layer

def init_layers_with_linear_mean_functions(X, Y, Z, kernels, D_Y):
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
        q_sqrt = np.eye(M)[:, :, None] * np.ones((1, 1, dim_out))

        layers.append(Layer(kern, q_mu, q_sqrt, Z_running, mean_function))

        Z_running = Z_running.dot(W)
        X_running = X_running.dot(W)

    return ParamList(layers)

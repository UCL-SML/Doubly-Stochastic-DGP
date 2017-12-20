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

import sys, os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socket

import itertools

from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF, White
from gpflow.training import AdamOptimizer

import gpflow_monitor

from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from scipy.special import logsumexp

from doubly_stochastic_dgp.dgp import DGP
from datasets import Datasets
datasets = Datasets()

results_path = '/vol/bitbucket/hrs13/tmp_results_{}/'.format(socket.gethostname())

dataset_name = str(sys.argv[1])
L = int(sys.argv[2])
split = int(sys.argv[3])


iterations = 10000
log_every = 100
minibatch_size = 10000



data = datasets.all_datasets[dataset_name].get_data(split=split)
X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]

print('############################ {} L={} split={}'.format(dataset_name, L, split))
print('N: {}, D: {}, Ns: {}'.format(X.shape[0], X.shape[1], Xs.shape[0]))

Z = kmeans2(X, 100, minit='points')[0]

D = X.shape[1]

kernels = []
for l in range(L):
    kernels.append(RBF(D))

for kernel in kernels[:-1]:
    kernel += White(D, variance=2e-6)

mb = minibatch_size if X.shape[0] > minibatch_size else None
model = DGP(X, Y, Z, kernels, Gaussian(), num_samples=1, minibatch_size=mb)

# start the inner layers almost deterministically
for layer in model.layers[:-1]:
    layer.q_sqrt = layer.q_sqrt.value * 1e-5
model.likelihood.variance = 0.05

global_step = tf.Variable(0, trainable=False, name="global_step")
model.enquire_session().run(global_step.initializer)

s = "{}/{}_L{}_split{}".format(results_path, dataset_name, L, split)
fw = tf.summary.FileWriter(os.path.join(s.format(dataset_name, L)),
                           model.enquire_session().graph)

opt_method = gpflow_monitor.ManagedOptimisation(model, AdamOptimizer(0.01), global_step)

its_to_print = (x * log_every for x in itertools.count())

opt_method.tasks += [
    gpflow_monitor.PrintTimings(its_to_print, gpflow_monitor.Trigger.ITER),
    gpflow_monitor.ModelTensorBoard(its_to_print, gpflow_monitor.Trigger.ITER,
                                    model, fw),
    gpflow_monitor.LmlTensorBoard(its_to_print, gpflow_monitor.Trigger.ITER,
                                  model, fw, verbose=False),
    gpflow_monitor.StoreSession(its_to_print, gpflow_monitor.Trigger.ITER,
                                model.enquire_session(), (s+'/checkpoints').format(dataset_name, L))
]

class TestTensorBoard(gpflow_monitor.ModelTensorBoard):
    def __init__(self, sequence, trigger: gpflow_monitor.Trigger, model, file_writer, Xs, Ys):
        super().__init__(sequence, trigger, model, file_writer)
        self.Xs = Xs
        self.Ys = Ys
        self._full_test_err = tf.placeholder(tf.float64, shape=())
        self._full_test_nlpp = tf.placeholder(tf.float64, shape=())

        self.summary = tf.summary.merge([tf.summary.scalar("test_rmse", self._full_test_err),
                                         tf.summary.scalar("test_nlpp", self._full_test_nlpp)])

    def _event_handler(self, manager):
        minibatch_size = 1000
        S = 100
        means, vars = [], []
        for mb in range(-(-len(Xs) // minibatch_size)):
            m, v = model.predict_y(Xs[mb * minibatch_size:(mb + 1) * minibatch_size, :], S)
            means.append(m)
            vars.append(v)
        mean_SND = np.concatenate(means, 1)
        var_SDN = np.concatenate(vars, 1)

        mean_ND = np.average(mean_SND, 0)

        test_err = np.average(Y_std * np.mean((Ys - mean_ND) ** 2.0) ** 0.5)
        test_nll_ND = logsumexp(norm.logpdf(Ys * Y_std, mean_SND * Y_std, var_SDN ** 0.5 * Y_std), 0, b=1 / float(S))
        test_nll = np.average(test_nll_ND)

        summary, step = model.enquire_session().run([self.summary, global_step],
                                                    feed_dict={self._full_test_err: test_err,
                                                               self._full_test_nlpp: test_nll})
        self.file_writer.add_summary(summary, step)



opt_method.tasks.append(TestTensorBoard(its_to_print, gpflow_monitor.Trigger.ITER,
                                        model, fw, Xs, Ys))




opt_method.minimize(maxiter=iterations)
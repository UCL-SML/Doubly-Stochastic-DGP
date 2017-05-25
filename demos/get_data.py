# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:01:24 2017

@author: hughsalimbeni
"""


import pandas
import numpy as np

from subprocess import Popen
import os

data_path = '../data/' # or somewhere else

SEED = 0  # change (by more than 20) for different splits
PROPORTION_TRAIN = 0.9

def download(name, data_path=data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    print 'Downloading file: {}'.format(name)
    download_url = 'https://hrs13publicdata.blob.core.windows.net/publicdata/'
    s1 = 'cd {}'.format(data_path)
    s2 = 'wget {}{}.gz'.format(download_url, name)
    s3 = 'gzip -d {}.gz'.format(name)
    s = '{}\n{}\n{}\n'.format(s1, s2, s3)
    proc = Popen(s, shell=True)
    proc.wait()
    assert os.path.isfile(data_path + name), 'something went wrong downloading'
    print 'Downloaded file: {}'.format(name)


def make_split(X_full, Y_full, split):
    N = X_full.shape[0]
    n = int(N * PROPORTION_TRAIN)
    ind = np.arange(N)    
    
    np.random.seed(split + SEED) 
    np.random.shuffle(ind)
    train_ind = ind[:n]
    test_ind= ind[n:]
    
    X = X_full[train_ind]
    Xs = X_full[test_ind]
    Y = Y_full[train_ind]
    Ys = Y_full[test_ind]
    
    return X, Y, Xs, Ys

def get_mnist_data(data_path=data_path):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_path+'MNIST_data/', one_hot=False)

    X, Y = mnist.train.next_batch(mnist.train.num_examples)
    Xval, Yval = mnist.validation.next_batch(mnist.validation.num_examples)
    Xtest, Ytest = mnist.test.next_batch(mnist.test.num_examples)

    Y, Yval, Ytest = [np.array(y, dtype=int) for y in [Y, Yval, Ytest]]

    X = np.concatenate([X, Xval], 0)
    Y = np.concatenate([Y, Yval], 0)

    return X, Y, Xtest, Ytest


def get_regression_data(name, split, data_path=data_path):
    path = '{}{}.csv'.format(data_path, name)

    if not os.path.isfile(path):
        download(name +'.csv', data_path=data_path)
        
    data = pandas.read_csv(path, header=None).values

    if name in ['energy', 'naval']:
        # there are two Ys for these, but take only the first
        X_full = data[:, :-2]
        Y_full = data[:, -2]
    else:
        X_full = data[:, :-1]
        Y_full = data[:, -1]
        
    
    X, Y, Xs, Ys = make_split(X_full, Y_full, split)
    
    ############# whiten inputs 
    X_mean, X_std = np.average(X, 0), np.std(X, 0)+1e-6
    
    X = (X - X_mean)/X_std
    Xs = (Xs - X_mean)/X_std
    
    return  X, Y[:, None], Xs, Ys[:, None]   


def get_taxi_data(chunk, data_path=data_path):
    file_name = 'taxi_data_shuffled_{}.csv'.format(chunk)
    path = data_path + file_name
    if not os.path.isfile(path):
        download(file_name, data_path=data_path)
    return pandas.read_csv(path, header=None).values
    
def get_taxi_stats(data_path=data_path):
    file_name = 'taxi_data_stats.p'
    path = data_path + file_name
    if not os.path.isfile(path):
        download(file_name, data_path=data_path)

    import pickle
    stats = pickle.load(open(path, 'r'))
    sum_X = stats['sum_X']
    sum_X2 = stats['sum_X2']
    n = float(stats['n'])
    X_mean = sum_X / n
    X_std = ((sum_X2 - (sum_X**2)/n)/(n-1))**0.5

    X_mean = np.reshape(X_mean, [1, -1])
    X_std = np.reshape(X_std, [1, -1])
    
    return X_mean[:, :-1], X_std[:, :-1] # these included the target, but we don't normalize that here
        



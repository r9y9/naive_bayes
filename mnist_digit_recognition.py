# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from naive_bayes import GaussianNaiveBayes
from pylab import *
import cPickle
import gzip
import os

def load_mnist_dataset(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

def create_2D_images_horizontal(x, w, h):
    N = x.shape[0]
    for n in range(N):
        subplot_instance = subplot(1, N, n)
        subplot_instance.tick_params(labelleft='off',labelbottom='off')
        reshape_data = x[n].reshape(w, h)
        create_2D_image(reshape_data)

def create_2D_image(x):
    row, col = x.shape
    a = arange(col+1)
    b = arange(row+1)
    a, b = meshgrid(a, b)
    imshow(x)

def mnist_digit_recognition():
    train_set, valid_set, test_set = load_mnist_dataset("mnist.pkl.gz")
    n_labels = 10 # 1,2,3,4,5,6,7,9,0
    n_features = 28*28
    mnist_model = GaussianNaiveBayes(n_labels, n_features)
    mnist_model.train(train_set[0], train_set[1])
    [mean, var], pi = mnist_model.get_parameters()

    # visualization of learned means
    create_2D_images_horizontal(mean, w=28, h=28)
    show()

    test_data, labels = test_set
    # slice
    limit = 50
    test_data, labels = test_data[:limit], labels[:limit]
    results = np.arange(limit, dtype=np.int)
    for n in range(limit):
        results[n] = mnist_model.classify(test_data[n])
        print "%d : predicted %s, correct %s" % (n, results[n], labels[n])
    # results = mnist_model.classify(test_data)

    print "recognition rate: ", (results == labels).mean()
        
if __name__=="__main__":
    mnist_digit_recognition()

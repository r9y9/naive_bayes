# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.stats
import cPickle

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes implementation
    
    public methods:
    classify
    train
    negative_log_likelihood
    save
    load
    get_parameters
    """
    def __init__(self, n_labels, n_features):
        self.n_labels = np.array(n_labels)
        self.n_features = np.array(n_features)
        self.mean = np.zeros((n_labels, n_features), dtype=np.float)
        self.var = np.zeros((n_labels, n_features), dtype=np.float)
        self.pi = np.zeros(n_labels, dtype=np.float)
            
    def classify(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("The input should be numpy.ndarray");
        # n-samles
        if len(data.shape) == 2:
            if data.shape[1] != self.n_features:
                raise ValueError("The input does not have dimension %s" % self.n_features)
            return np.array([self.__classify_inner(x) for x in data])
        # 1 sample
        if data.shape[0] != self.n_features:
            raise ValueError("The input does not have dimension %s" % self.n_features)
        return self.__classify_inner(data)

    def train(self, data, labels):
        for d in data, labels:
            if not isinstance(d, np.ndarray):
                raise ValueError("The input should be numpy.ndarray");
        if labels.shape[0] != data.shape[0]:
            raise ValueError("The number of data and the number of labels should be same")
        if data.shape != (data.shape[0], self.n_features):
            raise ValueError("The input does not have dimension %s" % self.n_features)

        N = data.shape[0] # the number of training data
        N_l = np.array([(labels == y).sum() for y in range(self.n_labels)], dtype=np.float) # count for each label

        # udpate mean of Gaussian
        for y in range(self.n_labels):
            sum = np.sum(data[n] if labels[n] == y else 0.0 for n in range(N))
            self.mean[y] = sum / N_l[y]

        # update variance of Gaussian
        for y in range(self.n_labels):
            sum = np.sum((data[n] - self.mean[y])**2 if labels[n] == y else 0.0 for n in range(N))
            self.var[y] = sum / N_l[y]

        # update prior of labels
        self.pi = N_l / N;

    def negative_log_likelihood(self, data, labels):
        for d in data, labels:
            if not isinstance(d, np.ndarray):
                raise ValueError("The input should be numpy.ndarray");
        if labels.shape[0] != data.shape[0]:
            raise ValueError("The number of data and the number of labels should be same")
        # n-samles
        if len(data.shape) == 2:
            if data.shape[1] != self.n_features:
                raise ValueError("The input does not have dimension %s" % self.n_features)
            N = data.shape[0] # the number of data
            return np.sum([self.__negative_log_likelihood_inner(data[n], labels[n]) for n in range(N)])

        if data.shape[0] != self.n_features:
            raise ValueError("The input does not have dimension %s" % self.n_features)
        # 1-sample
        return self.__negative_log_likelihood_inner(data, labels)

    def save(self, filename="gaussian_naive_bayes.w"):
        f = file(filename, "wb")
        parameters = ([self.mean, self.var], self.pi)
        cPickle.dump(parameters, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, filename="gaussian_naive_bayes.w"):
        f = file(filename, "rw")
        ([self.mean, self.var], self.pi) = cPickle.load(f)
        self.n_labels = self.pi.shape[0]
        self.n_features = self.mean[0].shape[0]

    def get_parameters(self):
        return ([self.mean, self.var], self.pi)

    def __classify_inner(self, x):
        results = [self.__negative_log_likelihood_inner(x, y) for y in range(self.n_labels)]
        return np.argmin(results)

    def __negative_log_likelihood_inner(self, x, y):
        log_prior_y = -np.log(self.pi[y])
        log_posterior_x_given_y = -np.sum([self.__log_gaussian_wrap(x[d], self.mean[y][d], self.var[y][d]) for d in range(self.n_features)])
        return log_prior_y + log_posterior_x_given_y

    def __log_gaussian_wrap(self, x, mean, var):
        epsiron = 1.0e-5
        if var < epsiron:
            return 0.0;
        return scipy.stats.norm(mean, var).logpdf(x)


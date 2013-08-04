# -*- coding: utf-8 -*-
#!/usr/bin/python

from naive_bayes import GaussianNaiveBayes
import numpy as np
    
def naive_bayes_test():
    n_labels, n_features = 2, 2
    nb = GaussianNaiveBayes(n_labels, n_features)

    # prepare sample training data
    data1 = np.random.multivariate_normal([1,4], [[2,0],[0,2]], size=100)
    data2 = np.random.multivariate_normal([5,7], [[3,0],[0,1]], size=100)
    data = np.concatenate((data1, data2), axis=0)

    # prepare training label data
    labels = np.concatenate((np.array([0]*100), np.array([1]*100)), axis=0)
    print "correct labels"
    print labels

    # nb.load()
    nb.train(data, labels)
    # nb.save()

    results = nb.classify(data)
    print "predicted labels"
    print results
    
    print "recognition rate: ", (results == labels).mean()

if __name__=="__main__":
    naive_bayes_test()

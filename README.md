naive_bayes
===========

Naive Bayes implementation with digit recognition example

## Requirement
- numpy
- scipy
- matplotlib

## Synthetic data example
	$ git clone https://github.com/r9y9/naive_bayes && cd naive_bayes
	$ python naive_bayes_test.py

## Digit recognition using MNIST dataset
	$ python mnist_digit_recognition.py

### After training
mean of Gaussians
![](mnist_mean_of_gaussian.png)

### Result
0.7634 (7634/10000)

Note that it spends about an hour to test 10000 exapmles in this simple implementation.

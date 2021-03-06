import numpy as np
class Perceptron(object):

  def __init__(self, eta=0.01, n_iter=10):
    # eta is the Learning rate, sometimes called alpha
    # should be between 0.0 and 1.0
    self.eta = eta
    # n_iter is number of iterations
    self.n_iter = n_iter
  
  def fit(self, X, y):
    # w_ Are the weights for the neurons
    self.w_ = np.zeros(1 + X.shape[1])
    self.errors_ = []

    for _ in range(self.n_iter):
      errors = 0
      for xi, target in zip(X, y):
        update = self.eta * (target - self.predict(xi))
        self.w_[1:] += update * xi
        self.w_[0] += update
        errors += int(update != 0.0)
      self.errors_.append(errors)
    return self

  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def predict(self, X):
    return np.where(self.net_input(X) >= 0.0, 1, -1)

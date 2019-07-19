# -*- coding:utf-8 -*-

import numpy as np
from tqdm import trange


class SoftmaxRegressor:
    def __init__(self, num_labels, lr, penalty="l2", gamma=0, fit_intercept=False):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.W = None
        self.b = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.num_labels = num_labels
        self.lr = lr

    def __call__(self, X, y):

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        if self.W is None:
            self.W = np.random.randn(X.shape[1], self.num_labels)
        if self.b is None:
            self.b = np.zeros((1, self.num_labels))
        # l_prev = np.inv
        # for iter_num in trange(int(max_iter)):
        y_pred = self._softmax(np.dot(X, self.W) + self.b)
        loss = self._NLL(X, self._one_hot(y), y_pred)
        return y_pred, np.mean(loss)
            # # early stop
            # if (np.mean(loss) - l_prev) < tol:
            #     return
            # # print
            # if iter_num % log_iter == 0:
            #     print("iter : {} and the loss : {}".format(iter, np.mean(loss)))

            # # gradient descent
            # d_W, d_b = self._NLL_grad(X, y, y_pred)
            # self.W -= lr * d_W
            # self.b -= lr * d_b

    def backward(self, X, y, y_pred):
        d_W, d_b = self._NLL_grad(X, self._one_hot(y), y_pred)
        self.W -= self.lr * d_W
        self.b -= self.lr * d_b

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        y_pred = self._softmax(np.dot(X, self.W) + self.b)
        return np.argmax(y_pred, axis=0)

    def save(self, path):
        np.save(path, self.W, self.b)
        print('Successfully save the wb to {}'.format(path))

    def load(self, path):
        npzfile = np.load(path)
        self.W = npzfile['arr_0']
        self.b = npzfile['arr_1']

    def _NLL(self, X, y, y_pred):
        N = X.shape[0]
        order = 2 if self.penalty == 'l2' else 1
        nll = self._cross_entropy(y_pred, y)
        penalty = 0.5 * self.gamma * np.linalg.norm(self.W, ord=order)
        return nll + penalty / N

    def _NLL_grad(self, X, y, y_pred):
        N = X.shape[0]
        d_penalty_W = self.gamma * self.W if self.penalty == 'l2' \
            else self.gamma * np.sign(self.W)
        # print()
        d_penalty_b = self.gamma * self.b if self.penalty == 'l2' \
            else self.gamma
        d_W = (np.dot(X.T, y_pred - y) + d_penalty_W) / N
        d_b = ((np.sum(y_pred - y, axis=0) + d_penalty_b) / N)
        return d_W, d_b

    def _one_hot(self, y):
        """
        [0, 1, 2] => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        one_hot = np.zeros((y.shape[0], self.num_labels))
        one_hot[np.arange(y.shape[0]), y.T] = 1
        return one_hot

    def _cross_entropy(self, scores, y_true):
        """
        loss = -1 / N * /sum y * log(y^)
        """
        loss = -(1 / scores.shape[0]) * np.sum(y_true * np.log(scores))
        return loss

    def _softmax(self, scores):
        """
        scores = W_{T}X + b which the shape is [1, n_labels]
        """
        return np.exp(scores) / sum(np.exp(scores))

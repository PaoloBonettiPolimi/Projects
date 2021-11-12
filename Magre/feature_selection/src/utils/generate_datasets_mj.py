import numpy as np
from numpy.random import multivariate_normal, normal
np.random.seed(123)


def generateBinary(size):
    """ Given y=-1, (X1, ... X10) are N(0, I_10),
        Given y=1, (X1, ... X4) are standard normal conditioned on
        9 \leq \sum_{j=1}^4 X_j^2 \leq 16; (X5, ... X10) are N(0, I_6)"""

    def _getConditioned():
        vals = multivariate_normal(np.zeros(4), np.identity(4))
        squared = vals ** 2
        if np.sum(squared) > 9 and np.sum(squared) < 16:
            return vals
        return _getConditioned()

    positive = []
    negative = []
    for i in range(size // 2):
        positive.append(multivariate_normal(np.zeros(10), np.identity(10)))
        neg = np.concatenate([_getConditioned(), multivariate_normal(
            np.zeros(6), np.identity(6))])
        negative.append(neg)

    out = np.vstack(positive + negative)
    labels = np.array([1] * (size // 2) + [-1] * (size // 2))
    return out, labels


def generateXOR(size):
    """3 dimensional XOR as a 4 way classification"""
    X = np.random.randn(size, 10)
    Y = np.zeros(size)
    splits = np.linspace(0, size, num=8+1, dtype=int)
    signals = [
        [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [1, -1, 1]]
    for i in range(8):
        X[splits[i]:splits[i+1], :3] += np.array([signals[i]])
        Y[splits[i]:splits[i+1]] = i // 2

    perm_inds = np.random.permutation(size)
    X, Y = X[perm_inds], Y[perm_inds]
    return X, Y


def generateNonLinearRegression(size):
    """ y = -2 sin(2x1) + max(x2, 0), + x3 + exp(-x4) + eps """
    def foo(x):
        eps = normal()
        return -2 * np.sin(2*x[0]) + np.max([x[1], 0]) + x[2] + np.exp(-x[3]) + eps

    X = []
    Y = []
    for i in range(size):
        x = multivariate_normal(np.zeros(10), np.identity(10))
        X.append(x)
        Y.append(foo(x))

    return np.vstack(X), np.array(Y)


def generateIdenticalFeatures(size):
    """
    y = -2 sin(2x1) + max(x2, 0), + x3 + exp(-x4) + eps
    x5 = -x1, x6 = -x2, x7 = -x3, x8 = -x4
    """
    def foo(x):
        eps = normal()
        return -2 * np.sin(2*x[0]) + np.max([x[1], 0]) + x[2] + np.exp(-x[3]) + eps

    X = []
    Y = []
    for i in range(size):
        x = multivariate_normal(np.zeros(4), np.identity(4))
        X.append(np.hstack([x, -x]))
        Y.append(foo(x))

    return np.vstack(X), np.array(Y)


def genereateDependentFeatures(size):
    """
    y = -2 sin(2x1) + max(x2, 0), + x3 + exp(-x4) + eps
    x5 = x1 + x2, x6 = x2 * x3, x7 = x4 - x1, x8 = x1
    """
    def foo(x):
        eps = normal()
        return -2 * np.sin(2*x[0]) + np.max([x[1], 0]) + x[2] + np.exp(-x[3]) + eps

    X = []
    Y = []
    for i in range(size):
        x = np.zeros(8)
        x[:4] = multivariate_normal(np.zeros(4), np.identity(4))
        x[4] = x[0] + x[1]
        x[5] = x[1] * x[2]
        x[6] = x[3] - x[0]
        x[7] = x[0]
        X.append(x)
        Y.append(foo(x))

    return np.vstack(X), np.array(Y)

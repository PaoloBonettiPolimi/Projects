"""
Generate synthetic datasets to test different stopping criteria.
"""
import numpy as np
from scipy.stats import multivariate_normal, norm
from sklearn.utils import shuffle

NONLINEAR_BASE_FUNCS = [
    np.sin,
    np.cos,
    lambda x: np.exp(-x),
    lambda x: max(x, 0),
]

NUM_BASE_FUNCS = len(NONLINEAR_BASE_FUNCS)

ALPHA = 3


def applyFuncs(X, funcs):
    """
    Applies a list of functions to a list of values
    """
    out = []
    for val, func in zip(X, funcs):
        out.append(func(val))
    return np.array(out)


def getFeatures(numFeatures, numSamples, numUsefulFeatures=-1):
    if numUsefulFeatures <= 0:
        numUsefulFeatures = np.random.randint(3, numFeatures)
    X = multivariate_normal.rvs(
        np.zeros(numFeatures), np.identity(numFeatures), numSamples)

    return {
        'X': X,
        'numUsefulFeatures': numUsefulFeatures}


def generateRegression(numDatasets, numFeatures, numSamples):
    """
    Generate a regression dataset:
    y_i = \sum_{j=1}^K \Phi_i(x_ij)

    - K ~ Uniform(3, numFeatures)
    - Phi_i is randomly selected among 4 nonlinear base function and other
      linear ones.
    - x_i is a vector with numFeatures components
    """
    out = []
    for index in range(numDatasets):
        data = getFeatures(numFeatures, numSamples)
        X = data['X']
        base_funcs = NONLINEAR_BASE_FUNCS.copy()
        if data['numUsefulFeatures'] > NUM_BASE_FUNCS:
            base_funcs.extend(
                [lambda x: x] * (NUM_BASE_FUNCS - data['numUsefulFeatures']))
        funcs = shuffle(base_funcs)[:data['numUsefulFeatures']]
        Y = np.zeros(numSamples)
        for i in range(numSamples):
            Y[i] = np.sum(applyFuncs(X[i, :], funcs)) + norm.rvs()
        data['Y'] = Y
        out.append(data)

    return out


def generateBinaryProblem(numFeatures, numSamples, numUseful):
    def _getConditioned(k, acceptFunc):
        while True:
            vals = multivariate_normal.rvs(np.zeros(k), np.identity(k))
            if acceptFunc(k, vals):
                return vals

    positive = []
    negative = []

    positiveAcceptFunc = lambda k, x: 3*(k-2) < np.sum(x**2) < 4*k
    negativeAcceptFunc = lambda k, x: np.sum(x**2) < 3.5*(k-2)

    if numUseful <= 0:
        numUseful = np.random.randint(3, 12)
    numUseless = numFeatures - numUseful
    zeros = np.zeros(numUseless)
    eye = np.identity(numUseless)
    for i in range(numSamples // 2):
        print("\rNum useful features:{0} row: {1}".format(numUseful, i),
              flush=True, end=" ")
        negative.append(
            np.concatenate([
                _getConditioned(numUseful, negativeAcceptFunc),
                multivariate_normal.rvs(zeros, eye)]))

        positive.append(
            np.concatenate([
                _getConditioned(numUseful, positiveAcceptFunc),
                multivariate_normal.rvs(zeros, eye)]))

    out = np.vstack(positive + negative)
    labels = np.array([1] * (numSamples // 2) + [0] * (numSamples // 2))
    return {'X': out, 'Y': labels, 'numUsefulFeatures': numUseful}


def generateBinary(numDatasets, numFeatures, numSamples, numUseful=-1):
    """
    Generate a binary classification dataset
    Given y = 1, (X_1, ... X_k) are N(0, I_k) | 3*(k-2) <= \sum X_i^2 <= 4*k
    Given y = 0 (X_1, ... X_k) are N(0, I_k) | 0 < \sum X_i^2 < 3.5*(k-1)
    (X_k+1, ... ,X_n) are N(0, I)

    if numUseful < 0:
        k ~ Uniform(3, numFeatures)
    else:
        k = numUseful
    """
    if numDatasets == 1:
        return generateBinaryProblem(numFeatures, numSamples, numUseful)

    out = []
    for i in range(numDatasets):
        out.append(generateBinaryProblem(numFeatures, numSamples, numUseful))
    return out


def generateLinearRegressionProblem(numFeatures, numSamples, numUseful=-1):
    data = getFeatures(numFeatures, numSamples, numUseful)
    X = data['X']
    weights = norm.rvs(scale=3, size=numUseful)
    Y = X[:, :numUseful] * weights
    data['Y'] = Y
    return data


def generateLinearRegression(numDatasets, numFeatures, numSamples, numUseful=-1):
    if numDatasets == 1:
        return generateLinearRegressionProblem(
            numFeatures, numSamples, numUseful)

    out = []
    for i in range(numDatasets):
        out.append(generateLinearRegressionProblem(
            numFeatures, numSamples, numUseful))
    return out

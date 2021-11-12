"""
Estimate gradient fitting a radial basis function model.
f(x) = \sum w_i \phi(||x- x_i||)
Being || * || the euclidean norm.
f'(x) = \sum w_i \phi'(||x- x_i||) (x-x_i) / ||x-x_i||
"""
import numpy as np
from scipy.interpolate import Rbf


class RbfGradient(object):
    def _gaussD(self, r):
        return -2 / self.fitted.epsilon * r * self.fitted._h_gaussian(r)

    def _linearD(self, r):
        return 1

    def _cubicD(self, r):
        return 3 * r**2

    def _multiquadricD(self, r):
        return r / (self.fitted.epsilon ** 2 * self.fitted._h_multiquadric(r))

    @staticmethod
    def dist(x1, x2):
        return np.sqrt(((x1 - x2)**2).sum(axis=0))

    def __init__(self, function: str):
        self.function = function
        _derivatives = {
            'gaussian': self._gaussD,
            'linear': self._linearD,
            'cubic': self._cubicD,
            'multiquadric': self._multiquadricD}
        self.basis_derivative = _derivatives[function]

    def computeGradient(self, point, others):
        grad = np.zeros(len(point))
        for ind in range(others.shape[0]):
            center = others[ind, :]
            r = self.dist(point, center)
            grad += self.basis_derivative(r) / r * (point - center) * \
                self.fitted.nodes[ind]

        return grad

    def getGradient(self, points, observations):
        grads = np.zeros(points.shape)
        self.fitted = Rbf(
            *points.transpose(), observations, function=self.function)
        for ind in range(points.shape[0]):
            grads[ind, :] = np.absolute(self.computeGradient(
                points[ind, :], np.delete(points, ind, axis=0)))

        return np.mean(grads, axis=0)

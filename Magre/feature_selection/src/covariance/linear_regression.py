"""Estimate inf_g E[x_i | x-i] = E_{x-i}[Var[x_i | x-i]] using a linear
regression"""
import numpy as np
from sklearn.linear_model import Lasso


class LRVarianceEstimator(object):
    def __init__(self, smoothing=0.1):
        self.model = Lasso(smoothing, max_iter=100000)

    def _computeError(self, X, Y):
        preds = self.model.predict(X)
        err = preds - Y
        return np.mean(err**2)

    def estimateFeatureCovariance(self, points, feature: int):
        """Estimate E_{x-i}[Var[x_i | x-i]]
        param: feature -- the selected feature"""

        Y = points[:, feature]
        X = np.delete(points, feature, axis=1)
        self.model.fit(X, Y)
        return self._computeError(X, Y)

    def estimateCovariances(self, points):
        """Estimate E_{x-i}[Var[x_i | x-i]] for all the features"""
        nFeatures = points.shape[1]
        variances = np.zeros(nFeatures)
        for feature in range(nFeatures):
            variances[feature] = self.estimateFeatureCovariance(points, feature)

        return variances

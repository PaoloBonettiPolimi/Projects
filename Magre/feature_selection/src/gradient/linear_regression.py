import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression


class LinearRegressionGradient(object):
    """Estimate the gradient through Lasso linear regression"""

    def __init__(self, categorical=False, smoothing=0.0001):
        self.categorical = categorical

        if categorical:
            self.model = LogisticRegression(max_iter=100000)
        else:
            self.model = Lasso(smoothing, max_iter=100000)

    def getGradient(self, points, observations):
        self.model.fit(points, observations)
        grad = self.model.coef_
        if self.categorical and len(observations.shape) == 1:
            grad = grad[0]

        return np.absolute(grad)

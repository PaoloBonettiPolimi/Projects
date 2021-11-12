# @Author: mario
# @Date:   2018-12-21T09:53:52+01:00
# @Last modified by:   mario
# @Last modified time: 2018-12-21T10:25:02+01:00

import abc
import logging
import numpy as np
from multiprocessing import Pool


class FeatueSelector(metaclass=abc.ABCMeta):
    def __init__(self, miEstimator, classification=False,
                 X=None, Y=None, nproc=1):
        self.miEstimator = miEstimator
        self.classification = classification
        self.original_features = X
        self.responses = Y
        self.current_features = X
        self.nproc = nproc
        if (X is not None) and (Y is not None):
            self.setup()

    def setData(self, X, Y):
        self.original_features = X
        self.responses = Y
        self.current_features = X
        self.setup()

    def setup(self):
        self.nFeaturesOring = self.original_features.shape[1]
        self.Ymax = np.max(self.responses)
        self.idMap = {k: k for k in range(self.nFeaturesOring)}
        self.featureScore = -1
        self.deltaScore = -1
        self.prevScore = 0
        self.nFeatures = self.nFeaturesOring
        self.error = 0
        self.res = 0

    def selectFeatures(
            self, criterion="num_features", num_features=-1,
            max_error=-1, delta_score=-1, feature_score=-1):

        if criterion == "num_features":
            return self._selectKFeatures(num_features)

        elif criterion == "error":
            return self._selectOnError(max_error)

        elif criterion == "delta_score":
            return self._selectOnDeltaScore(delta_score)

        elif criterion == "feature_score":
            return self._selectOnFeatureScore(feature_score)

        logging.error('Invalid criterion: {0}'.format(criterion))
        return

    def scoreFeatures(self):
        if self.nproc > 1:
            return self._scoreFeatureParallel()
        else:
            return self._scoreFeatureSequential()

    def computeError(self):
        if self.classification:
            error = np.sqrt(self.res * 2)
        else:
            error = 2 * self.Ymax**2 * self.res
        return error

    @abc.abstractmethod
    def _selectKFeatures(self, num_features):
        pass

    @abc.abstractmethod
    def _selectOnError(self, max_error):
        pass

    @abc.abstractmethod
    def _selectOnDeltaScore(self, eps):
        pass

    @abc.abstractmethod
    def _selectOnFeatureScore(self, threshold):
        pass

# @Author: mario
# @Date:   2018-12-13T09:39:39+01:00
# @Last modified by:   mario
# @Last modified time: 2019-01-08T16:43:51+01:00
import tqdm
import logging
import numpy as np
from multiprocessing import Pool
from src.mutual_information.feature_selection import FeatueSelector


class BackwardFeatureSelector(FeatueSelector):
    def __init__(self, miEstimator, classification=False, X=None, Y=None, nproc=1):
        super().__init__(miEstimator, classification, X, Y, nproc)

    def _removeWorstId(self, id):
        self.current_features = np.delete(self.current_features, id, axis=1)
        print("Removing original feature: {0}".format(self.idMap[id]))
        for k, v in list(self.idMap.items())[:-1]:
            if k >= id:
                self.idMap[k] = self.idMap[k+1]
        self.idMap.pop(max(self.idMap))

    def _selectKFeatures(self, num_features):
        while num_features < self.nFeatures:
            scores = self.scoreFeatures()
            worstId = scores[0][0]
            self._removeWorstId(worstId)
            self.nFeatures -= 1

        return set(self.idMap.values())

    def _selectOnError(self, max_error):
        error = 0.0
        while error < max_error and len(self.idMap) > 1:
            scores = self.scoreFeatures()
            print("Score: {0}".format(scores[0][1]))
            self.res += max(scores[0][1], 0)
            error = self.computeError()
            print("Error: {0}".format(error))
            if error >= max_error:
                return set(self.idMap.values())

            print("Worst ID: {0}".format(scores[0][0]))
            self._removeWorstId(scores[0][0])

        return set(self.idMap.values())

    def _selectOnDeltaScore(self, eps):
        """
        Keep removing features until there is a knee in the MI
        """
        while self.deltaScore < eps and len(self.idMap) > 1:
            scores = self.scoreFeatures()
            deltaScore = self.prevScore - scores[0][1]
            if deltaScore >= eps:
                return set(self.idMap.values())

            self._removeWorstId(scores[0][0])
            self.prevScore = scores[0][1]
        return set(self.idMap.values())

    def _selectOnFeatureScore(self, threshold):
        """
        Keep removing features until we found a feature with an MI greter
        than threshold
        """
        while self.featureScore < threshold and len(self.idMap) > 1:
            scores = self.scoreFeatures()
            self.featureScore = scores[0][1]
            if self.featureScore >= threshold:
                return set(self.idMap.values())

            self._removeWorstId(scores[0][0])
        return set(self.idMap.values())

    def _scoreFeatureParallel(self):
        args = []
        print("Number of features to score: {0}".format(self.current_features.shape[1]))
        for i in range(self.current_features.shape[1]):
            args.append(
                (self.current_features[:, i],
                 self.responses,
                 np.delete(self.current_features, i, axis=1)))
        with Pool(self.nproc) as p:
            scores = p.starmap(self.miEstimator.estimateConditionalMI, args)
        print("Finished scoring")
        scores = np.array(scores)
        featureAndScores = list(zip(range(len(scores)), scores))
        return sorted(featureAndScores, key=lambda x: x[1])

    def _scoreFeatureSequential(self):
        scores = np.zeros(self.current_features.shape[1])
        for i in tqdm.tqdm(range(self.current_features.shape[1])):
            scores[i] = self.miEstimator.estimateConditionalMI(
                self.current_features[:, i], self.responses,
                np.delete(self.current_features, i, axis=1))

        featureAndScores = list(zip(range(len(scores)), scores))
        return sorted(featureAndScores, key=lambda x: x[1])

import argparse
import numpy as np
import pickle
from sklearn.utils import shuffle

import src.utils.generate_datasets as utils


def saveToFile(points, labels, basefile):
    size = points.shape[0]
    fileName = "{0}_{1}.pickle".format(basefile, size)
    with open(fileName, 'wb') as fp:
        pickle.dump({'X': points, 'Y': labels}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_basefile", type=str, default="")
    parser.add_argument("--regression_basefile", type=str, default="")
    parser.add_argument("--xor_basefile", type=str, default="")
    parser.add_argument("--identical_features_basefile", type=str, default="")
    parser.add_argument("--dependent_features_basefile", type=str, default="")
    args = parser.parse_args()

    if args.binary_basefile:
        for i in range(1, 11):
            currPoints, currLabels = utils.generateBinary(10)
            if i == 1:
                points, labels = currPoints, currLabels
            else:
                points = np.vstack((points, currPoints))
                labels = np.concatenate((labels, currLabels))
            saveToFile(points, labels, args.binary_basefile)

    if args.regression_basefile:
        for i in range(1, 11):
            currPoints, currLabels = utils.generateNonLinearRegression(10)
            if i == 1:
                points, labels = currPoints, currLabels
            else:
                points = np.vstack((points, currPoints))
                labels = np.concatenate((labels, currLabels))
            saveToFile(points, labels, args.regression_basefile)

    if args.xor_basefile:
        for i in range(1, 11):
            currPoints, currLabels = utils.generateXOR(10)
            if i == 1:
                points, labels = currPoints, currLabels
            else:
                points = np.vstack((points, currPoints))
                labels = np.concatenate((labels, currLabels))
            saveToFile(points, labels, args.xor_basefile)

    if args.identical_features_basefile:
        for i in range(1, 11):
            currPoints, currLabels = utils.generateIdenticalFeatures(10)
            if i == 1:
                points, labels = currPoints, currLabels
            else:
                points = np.vstack((points, currPoints))
                labels = np.concatenate((labels, currLabels))
            saveToFile(points, labels, args.identical_features_basefile)

    if args.dependent_features_basefile:
        for i in range(1, 11):
            currPoints, currLabels = utils.genereateDependentFeatures(10)
            if i == 1:
                points, labels = currPoints, currLabels
            else:
                points = np.vstack((points, currPoints))
                labels = np.concatenate((labels, currLabels))
            saveToFile(points, labels, args.dependent_features_basefile)

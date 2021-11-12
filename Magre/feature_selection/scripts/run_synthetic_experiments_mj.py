import argparse
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from src.mutual_information.mutual_information import MixedRvMiEstimator
from src.mutual_information.backward_feature_selection import BackwardFeatureSelector
from src.mutual_information.forward_feature_selection import ForwardFeatureSelector
from mj_py3.core.ccm import ccm


def runOnDataset(bSelector, fSelector, X, Y, type_y, epsilon, correct: set):
    accuracies = []
    nTarget = len(correct)

    # Backward feature selection
    selected = bSelector.selectFeatures(X, Y, num_features=nTarget)
    accuracies.append(len(selected.intersection(correct)) / len(correct))

    # forward feature selection
    selected = fSelector.selectFeatures(X, Y, num_features=nTarget)
    accuracies.append(len(selected.intersection(correct)) / len(correct))

    # Conditional Covariance Minimzation
    scored = ccm(X, Y, nTarget, type_y, epsilon)
    selected = set(scored[:nTarget])
    accuracies.append(len(selected.intersection(correct)) / len(correct))
    return tuple(accuracies)


def runExperiment(basepath, name, type_y, correct: set):
    accuracies = []
    epsilon = 0.001 if name in ['binary', 'xor'] else 0.1
    for size in np.arange(10, 101, 10):
        print('Running Experiment {0}, size: {1}'.format(name, size))
        fileName = fileName = os.path.join(
            basepath, "{0}/{0}_{1}.pickle".format(name, size))
        with open(fileName, 'rb') as fp:
            data = pickle.load(fp)

        X = data['X']
        Y = data['Y']
        k = 3
        if name == "regression":
            k = 5
        # if name == "xor":
        #     k = 2 if size < 30 else 3
        #     Xscaler = MinMaxScaler()
        #     Yscaler = MinMaxScaler()
        #     Xscaler.fit(X)
        #     Yscaler.fit(Y.reshape(-1, 1))
        #     X = Xscaler.transform(X)
        #     Y = Yscaler.transform(Y.reshape(-1, 1))[:, 0]
        # if name == "binary":
        #     k = 4

        miNN = MixedRvMiEstimator(k, np.inf)
        bSelector = BackwardFeatureSelector(miNN)
        fSelector = ForwardFeatureSelector(miNN)
        accuracies.append(
            runOnDataset(bSelector, fSelector, X, Y, type_y, epsilon, correct))
    return accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basepath", type=str, default="data/synthetic/")
    parser.add_argument("--plot", type=str, default="")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    accuracies = {}

    accuracies['binary'] = runExperiment(
        args.basepath, 'binary', 'binary', {0, 1, 2, 3})

    accuracies['xor'] = runExperiment(
        args.basepath, 'xor', 'categorical', {0, 1, 2})

    accuracies['regression'] = runExperiment(
        args.basepath, 'regression', 'real-valued', {0, 1, 2, 3})
    #
    # accuracies['identical_features'] = runExperiment(
    #     args.basepath, 'identical_features', 'real-valued', {0, 1, 2, 3})
    #
    # accuracies['dependent_features'] = runExperiment(
    #     args.basepath, 'dependent_features', 'real-valued', {0, 1, 2, 3})

    with open(args.output, 'wb') as fp:
        pickle.dump(accuracies, fp)

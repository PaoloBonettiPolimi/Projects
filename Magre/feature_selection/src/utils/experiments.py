# @Author: mario
# @Date:   2018-12-13T10:04:45+01:00
# @Last modified by:   mario
# @Last modified time: 2018-12-21T11:12:30+01:00



import logging
import numpy as np
import pickle
from collections import defaultdict
from functools import partial
from itertools import product
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

from src.mutual_information.mutual_information import MixedRvMiEstimator
from src.mutual_information.backward_feature_selection import BackwardFeatureSelector
from src.mutual_information.forward_feature_selection import ForwardFeatureSelector
from src.utils.result import Result

K_GRID_META = [4, 6, 8, 10, 12]
K_GRID_MJ = [1, 2, 4, 6, 8]
ERROR_GRID_REGRESSION = [1, 2.5, 5, 7.5, 10, 12.5, 15]
ERROR_GRID_BINARY = [0.05, 0.5, 1.0, 2.0, 3.0, 4.0]
DELTASCORE_GRID = [0.05, 0.1, 0.25, 1, 2, 5]
FEATURESCORE_GRID = [0.1, 0.25, 0.5, 1, 2, 5]

EPS = 0.05


def selectKFeatures(selector, X, Y, k):
    return selector.selectFeatures(
        X, Y, criterion="num_features", num_features=k)


def selectOnError(selector, X, Y, max_error):
    return selector.selectFeatures(
        X, Y, criterion="error", max_error=max_error)


def selectOnDeltaScore(selector, X, Y, score):
    return selector.selectFeatures(
        X, Y, criterion="delta_score", delta_score=score)


def selectOnFeatureScore(selector, X, Y, score):
    return selector.selectFeatures(
        X, Y, criterion="feature_score", feature_score=score)


def runMetaOnDataset(binary: bool, data, mj=False):
    """
    Run fully meta-experiments, the metrics used are:
     - numSelectedRight
     - numSelectedWrong lower is better
     - svmAccuracy (0-1) for binary classification, (-inf, 1) for regression
       higher is better
     - lmAccuracy
    """
    miEstimator = MixedRvMiEstimator(3)
    out = defaultdict(list)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data['X'], data['Y'])

    usefulFeatures = list(range(data['numUsefulFeatures']))
    kgrid = K_GRID_MJ if mj else K_GRID_META
    kgrid = sorted(kgrid, reverse=True)
    selector = BackwardFeatureSelector(miEstimator, binary, Xtrain, Ytrain)
    for k in kgrid:
        res = Result('num_features', k, usefulFeatures, binary=binary)
        selected = selector.selectFeatures(
            criterion="num_features", num_features=k)
        res.addSelectionResult(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['num_features'].append(res)

    if binary:
        err_grid = ERROR_GRID_BINARY
    else:
        err_grid = ERROR_GRID_REGRESSION

    selector = BackwardFeatureSelector(miEstimator, binary, Xtrain, Ytrain)
    for err in err_grid:
        res = Result('error', err, usefulFeatures, binary=binary)
        selected = selector.selectFeatures(
            criterion="error", max_error=err)
        res.addSelectionResult(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['error'].append(res)

    for score in DELTASCORE_GRID:
        selector = BackwardFeatureSelector(miEstimator, binary, Xtrain, Ytrain)
        res = Result('delta_score', score, usefulFeatures, binary=binary)
        selected = selector.selectFeatures(
            criterion="delta_score", delta_score=score)
        res.addSelectionResult(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['delta_score'].append(res)

    selector = BackwardFeatureSelector(miEstimator, binary, Xtrain, Ytrain)
    for score in FEATURESCORE_GRID:
        res = Result('feature_score', score, usefulFeatures, binary=binary)
        selected = selector.selectFeatures(
            criterion="feature_score", feature_score=score)
        res.addSelectionResult(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['feature_score'].append(res)

    return out


def runAccuracyOnDataset(binary: bool, data, mj=True):
    """
    Run experiments to compare accuracy of different stopping criteria
    """
    miEstimator = MixedRvMiEstimator(3)
    selector = BackwardFeatureSelector(miEstimator, binary)
    out = defaultdict(list)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data['X'], data['Y'])

    k_grid = K_GRID_MJ if mj else K_GRID_META
    for k in k_grid:
        res = Result('num_features', k, binary=binary)
        selected = selectKFeatures(selector, Xtrain, Ytrain, k)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['num_features'].append(res)

    if binary:
        err_grid = ERROR_GRID_BINARY
    else:
        err_grid = ERROR_GRID_REGRESSION
    for err in err_grid:
        res = Result('error', err, binary=binary)
        selected = selectOnError(selector, Xtrain, Ytrain, err)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['error'].append(res)

    for score in DELTASCORE_GRID:
        res = Result('delta_score', score, binary=binary)
        selected = selectOnDeltaScore(selector, Xtrain, Ytrain, score)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['delta_score'].append(res)

    for score in FEATURESCORE_GRID:
        res = Result('feature_score', score, binary=binary)
        selected = selectOnFeatureScore(selector, Xtrain, Ytrain, score)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out['feature_score'].append(res)

    return out


def runMjDataset(binary: bool, data):
    """
    We know that the data was generated using only the first 4 features
    """
    data['weights'] = np.array([0.25] * 4)
    data['numUsefulFeatures'] = 4
    return runMetaOnDataset(binary, data, mj=True)


def runBinaryBig(data, nproc=1):
    """
    Runs the feature selection algorithm based on the error and on
    the number of features, evaluates the svm accuracy for each value of
    the grid
    """
    out = []
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data['X'], data['Y'])
    numSamplesGrid = list(range(250, 801, 50))
    func = partial(_binaryBigAux, Xtrain, Xtest, Ytrain, Ytest)
    with Pool(nproc) as p:
        out.extend(p.map(func, numSamplesGrid))
    return out


def _binaryBigAux(Xtrain, Xtest, Ytrain, Ytest, numSamples):
    out = []
    Xtrain = Xtrain[:numSamples, ]
    Ytrain = Ytrain[:numSamples, ]
    numNeighbors = [numSamples // x for x in [10, 20, 40]]
    print("Starting num samples: {0}".format(numSamples))
    for numNeigh in numNeighbors:
        miEstimator = MixedRvMiEstimator(numNeigh, nproc=1)
        selector = BackwardFeatureSelector(miEstimator, True, Xtrain, Ytrain)
        for err in [0.05, 0.5]:
            selected = selector.selectFeatures(
                criterion="error", max_error=err)
            res = Result('error', err, numNeighbors=numNeigh)
            res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
            res.numSamples = numSamples
            out.append(res)
    print("Finishing num samples: {0}".format(numSamples))
    return out


def runBinariesByFeatures(data, nproc=1):
    # datasets: a list of 5 dataset with the same number of useful features
    neighborsGrid = [3, 5, 7]
    out = []

    allDatasets = [dataset for datasets in data for dataset in datasets]
    iterable = list(product(neighborsGrid, allDatasets))
    print("Number of datasets to process: {0}".format(len(iterable)))
    for index in range(0, len(iterable), nproc):
        end = index + nproc
        print('Processing datasets {0}:{1}'.format(index, end))
        with Pool(nproc) as p:
            out.extend(p.map(_runOneDataset, iterable[index:end]))

    return out


def _runOneDataset(numNeighborsAndDataset):
    numNeighbors = numNeighborsAndDataset[0]
    dataset = numNeighborsAndDataset[1]
    kGrid = [20, 10, 5]
    errGrid = [0.05, 0.5, 1.0]
    deltaScoreGrid = [0.05, 0.15, 0.5]
    featureScoreGrid = [0.05, 0.15, 0.5]

    out = []
    numUseful = dataset['numUsefulFeatures']
    miEstimator = MixedRvMiEstimator(numNeighbors)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        dataset['X'], dataset['Y'])
    selector = BackwardFeatureSelector(
        miEstimator, True, Xtrain, Ytrain)
    for k in kGrid:
        logging.error("Selecting with k: {0}".format(k))
        res = Result(
            'num_features', k, numUseful=numUseful,
            numNeighbors=numNeighbors)
        selected = selector.selectFeatures(
            criterion="num_features", num_features=k)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out.append(res)

    selector = BackwardFeatureSelector(
        miEstimator, True, Xtrain, Ytrain)
    for err in errGrid:
        logging.error("Selecting with err: {0}".format(err))
        res = Result(
            'error', err, numUseful=numUseful,
            numNeighbors=numNeighbors)
        selected = selector.selectFeatures('error', max_error=err)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out.append(res)

    selector = BackwardFeatureSelector(
        miEstimator, True, Xtrain, Ytrain)
    for score in deltaScoreGrid:
        res = Result(
            'delta_score', score, numUseful=numUseful,
            numNeighbors=numNeighbors)
        selected = selector.selectFeatures(
            'delta_score', delta_score=score)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out.append(res)

    selector = BackwardFeatureSelector(
        miEstimator, True, Xtrain, Ytrain)
    for score in featureScoreGrid:
        res = Result(
            'feature_score', score, numUseful=numUseful,
            numNeighbors=numNeighbors)
        selected = selector.selectFeatures(
            'feature_score', feature_score=score)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out.append(res)
    return out


def runRealWord(datasets, binary=[], nproc=1, backward=True):
    """
    datasets: list of tuples (name, data)

    binary: a list of bool, one per dataset, indicating if it's a
    classification or regression problem.
    If empty, binary=True is assumed for all datasets
    """
    if not binary:
        binary = [True] * len(datasets)

    args = []
    for b, dataset in list(zip(binary, datasets)):
        args.append({
            'dataset': dataset, 'binary': b,
            'nproc': nproc, 'backward': backward})
    print("Number of datasets to process: {0}".format(len(args)))
    out = [_runOneReal(x) for x in args]
    return out


def _runOneReal(kwargs):
    out = []
    binary = kwargs['binary']
    name = kwargs['dataset'][0]
    data = kwargs['dataset'][1]
    print(data["X"].shape)
    numNeighbors = data["X"].shape[0] // 20

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        data['X'], data['Y'])

    if kwargs['backward']:
        miEstimator = MixedRvMiEstimator(numNeighbors, nproc=1)
        selector = BackwardFeatureSelector(
            miEstimator, True, Xtrain, Ytrain, nproc=kwargs["nproc"])
    else:
        miEstimator = MixedRvMiEstimator(numNeighbors, nproc=kwargs["nproc"])
        selector = ForwardFeatureSelector(
            miEstimator, True, Xtrain, Ytrain)

    grid = ERROR_GRID_BINARY if binary else ERROR_GRID_REGRESSION
    grid = sorted(grid)
    print("Starting dataset: {0}, num neighbors: {1}".format(name, numNeighbors))
    for err in grid:
        res = Result('error', err, numNeighbors=numNeighbors, name=name)
        selected = selector.selectFeatures('error', max_error=err)
        res.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)
        out.append(res)
    filename = "dataset_{0}_{1}_res.pickle".format(
        name, "backward" if kwargs["backward"] else "forward")
    with open(filename, "wb") as fp:
        pickle.dump(out, fp)
    print("Finished dataset: {0}, num neighbors: {1}".format(name, numNeighbors))
    return out

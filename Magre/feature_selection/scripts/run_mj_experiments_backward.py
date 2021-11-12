"""
Run our feature selection algorithm on MJ's datasets
"""

import argparse
import pickle

from src.utils.experiments import runAccuracyOnDataset, runMjDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_file", type=str,
        default="data/synthetic_mj/binary/binary_100.pickle")
    parser.add_argument(
        "--regression_file", type=str,
        default="data/synthetic_mj/regression/regression_100.pickle")
    parser.add_argument(
        "--dependent_file", type=str,
        default="data/synthetic_mj/dependent_features/dependent_features_100.pickle")
    parser.add_argument(
        "--identical_file", type=str,
        default="data/synthetic_mj/identical_features/identical_features_100.pickle")
    parser.add_argument(
        "--results_file", type=str, default="mj_results.pickle")
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    with open(args.binary_file, 'rb') as fp:
        binary = pickle.load(fp)

    with open(args.regression_file, 'rb') as fp:
        regression = pickle.load(fp)

    with open(args.dependent_file, 'rb') as fp:
        dependent = pickle.load(fp)

    with open(args.identical_file, 'rb') as fp:
        identical = pickle.load(fp)

    dependentStats = runAccuracyOnDataset(False, dependent, mj=True)
    identicalStats = runAccuracyOnDataset(False, identical, mj=True)
    regressionStats = runMjDataset(False, regression)
    binaryStats = runMjDataset(True, binary)

    with open(args.results_file, 'wb') as fp:
        out = {
            'binary': binaryStats,
            'regression': regressionStats,
            'dependent': dependentStats,
            'identical': identicalStats}
        pickle.dump(out, fp)

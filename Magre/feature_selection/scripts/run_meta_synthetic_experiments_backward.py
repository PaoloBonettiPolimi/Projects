import argparse
import pickle
from functools import partial
from multiprocessing import Pool

from src.utils.experiments import runMetaOnDataset


def runDatasets(datasets, binary: bool, nproc: int):
    func = partial(runMetaOnDataset, binary)
    out = []
    for index in range(0, len(datasets), nproc):
        start = index
        end = index + nproc
        print('Processing datasets {0}:{1}'.format(index, end))
        with Pool(nproc) as p:
            out.extend(p.map(func, datasets[start:end]))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_file", type=str, default="data/synthetic/binary.pickle")
    parser.add_argument(
        "--regression_file", type=str, default="data/synthetic/regression.pickle")
    parser.add_argument("--results_file", type=str, default="results.pickle")
    parser.add_argument("--nproc", type=int, default=1)
    args = parser.parse_args()

    with open(args.binary_file, 'rb') as fp:
        binary = pickle.load(fp)

    with open(args.regression_file, 'rb') as fp:
        regression = pickle.load(fp)

    binaryStats = runDatasets(binary, True, args.nproc)
    regressionStats = runDatasets(regression, False, args.nproc)

    with open(args.results_file, 'wb') as fp:
        pickle.dump({'binary': binaryStats, 'regression': regressionStats}, fp)

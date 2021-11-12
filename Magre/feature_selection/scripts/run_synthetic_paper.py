import argparse
import pickle

from src.utils.experiments import runBinaryBig, runBinariesByFeatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_big_data", type=str,
        default="data/synthetic/paper/binary_big.pickle")
    parser.add_argument(
        "--binary_big_res", type=str, default="binary_big_res.pickle")
    parser.add_argument(
        "--binaries_by_features_data", type=str,
        default="data/synthetic/paper/binaries_by_features.pickle")
    parser.add_argument(
        "--binaries_by_features_res", type=str,
        default="binaries_by_features_res.pickle")
    parser.add_argument("--nproc", type=int)
    args = parser.parse_args()

    print("----- BINARY BIG -----")

    if args.binary_big_data:
        with open(args.binary_big_data, 'rb') as fp:
            data = pickle.load(fp)

        results = runBinaryBig(data, args.nproc)
        with open(args.binary_big_res, 'wb') as fp:
            pickle.dump(results, fp)

    print("----- BINARIES BY FEATURES -----")
    if args.binaries_by_features_data:
        with open(args.binaries_by_features_data, 'rb') as fp:
            data = pickle.load(fp)

        results = runBinariesByFeatures(data, args.nproc)
        with open(args.binaries_by_features_res, 'wb') as fp:
            pickle.dump(results, fp)

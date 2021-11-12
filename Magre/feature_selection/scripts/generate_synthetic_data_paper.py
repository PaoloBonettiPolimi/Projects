import argparse
import pickle

from src.utils.generate_data import generateBinary, generateLinearRegression

USEFUL_FEATURES_GRID = [6, 9, 12, 15, 18, 21, 24]
NUM_FEATURES = 30

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary_big", type=str)
    parser.add_argument("--linear_regression", type=str)
    parser.add_argument("--binaries_by_features", type=str)
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    if args.binary_big:
        data = generateBinary(1, NUM_FEATURES, args.num_samples, 10)
        with open(args.binary_big, 'wb') as fp:
            pickle.dump(data, fp)

    if args.linear_regression:
        data = generateLinearRegression(1, NUM_FEATURES, args.num_samples, 10)
        with open(args.linear_regression, 'wb') as fp:
            pickle.dump(data, fp)

    if args.binaries_by_features:
        data = []
        for numUseful in USEFUL_FEATURES_GRID:
            data.append(generateBinary(
                5, NUM_FEATURES, args.num_samples, numUseful))

        with open(args.binaries_by_features, 'wb') as fp:
            pickle.dump(data, fp)

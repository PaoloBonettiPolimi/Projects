import argparse
import pickle

from src.utils.generate_data import generateRegression, generateBinary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_file", type=str, default='data/synthetic_v2/binary.pickle')
    parser.add_argument(
        "--regression_file", type=str,
        default='data/synthetic_v2/regression.pickle')
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_features", type=int, default=15)
    parser.add_argument("--num_datasets", type=int, default=50)
    args = parser.parse_args()

    regression = generateRegression(
        args.num_datasets, args.num_features, args.num_samples)

    with open(args.regression_file, 'wb') as fp:
        pickle.dump(regression, fp)

    binaryClassification = generateBinary(
        args.num_datasets, args.num_features, args.num_samples)

    with open(args.binary_file, 'wb') as fp:
        pickle.dump(binaryClassification, fp)

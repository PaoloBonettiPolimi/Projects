# @Author: mario
# @Date:   2018-12-13T18:27:49+01:00
# @Last modified by:   mario
# @Last modified time: 2018-12-21T10:49:03+01:00



import argparse
import glob
import os
import pickle

from src.utils.experiments import runRealWord

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--backward", default="T")
    parser.add_argument("--input_folder", type=str, default="data/real/")
    parser.add_argument("--output", type=str, default="real_data_res.pickle")
    parser.add_argument("--nproc", type=int)
    args = parser.parse_args()

    if args.dataset:
        with open(args.dataset, 'rb') as fp:
            data = pickle.load(fp)
        datasetName = args.dataset.split("/")[-1].strip(".pickle")
        out = runRealWord(
            [(datasetName, data)], nproc=args.nproc,
            backward=bool(args.backward))

    else:
        datasets = []
        filenames = glob.glob(os.path.join(args.input_folder, "*.pickle"))
        for fname in filenames:
            datasetName = fname.split("/")[-1].strip(".pickle")

            with open(fname, 'rb') as fp:
                data = pickle.load(fp)

            datasets.append((datasetName, data))

        results = runRealWord(datasets, nproc=args.nproc,
                              backward=bool(args.backward))
        with open(args.output, 'wb') as fp:
            pickle.dump(results, fp)

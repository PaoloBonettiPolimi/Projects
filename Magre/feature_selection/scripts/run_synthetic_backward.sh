#! /bin/bash

NPROC=${1}

python3 -m scripts.run_meta_synthetic_experiments_backward \
  --results_file meta_results_new_algo.pickle \
  --nproc ${NPROC}
#
# python3 -m scripts.run_mj_experiments_backward \
#   --results_file mj_results.pickle

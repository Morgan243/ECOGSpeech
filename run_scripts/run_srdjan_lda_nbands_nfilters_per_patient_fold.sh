#!/bin/bash
NUM_BANDS=(2 4 5)
N_FILTERS=(32)

for num_bands in "${NUM_BANDS[@]}"
do
  for n_filters in "${N_FILTERS[@]}"
  do
    ./run_per_patient_fold.sh "$@" --track-sinc-params --sn-n-bands=$num_bands --n-cnn-filters=$n_filters
  done
done
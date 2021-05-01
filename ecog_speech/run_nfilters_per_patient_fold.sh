#!/bin/bash

for n_filters in "${N_FILTERS[@]}"
do
  ./run_per_patient_fold.sh "$@" --track-sinc-params --n-cnn-filters=$n_filters
done
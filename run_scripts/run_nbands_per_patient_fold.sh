#!/bin/bash
NUM_BANDS=(1 3 5)
#N_FILTERS=(16 32)
#n_filters=16

for num_bands in "${NUM_BANDS[@]}"
do
    ./run_per_patient_fold.sh "$@" --track-sinc-params --sn-n-bands=$num_bands #--n-cnn-filters=$n_filters
done
#!/bin/bash
#NUM_BANDS=(1 2)
NUM_BANDS=(1 2 3 4 5)
#NUM_BANDS=(40 50 60 80 90)
LEARNING_RATES=(0.001)
N_EPOCHS=15
RESULT_DIR='../results'

mkdir -p $RESULT_DIR

for num_bands in "${NUM_BANDS[@]}"
do
  for lr in "${LEARNING_RATES[@]}"
  do
    echo --num-bands=$num_bands --learning-rate=$lr "$@"
    python experiments.py --result-dir=$RESULT_DIR \
                          --dataset='nww' \
                          --model-name='base-sn' \
                          --sn-n-bands=$num_bands \
                          --n-epochs=$N_EPOCHS \
                          --learning-rate=$lr "$@"
  done
done

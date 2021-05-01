#!/bin/bash

if [ $# -eq 0 ]
then
  SET_ID=19
  echo "Pass the set id - setting default: $SET_ID"
else
  SET_ID=$1
fi


if [[ $SET_ID == 19 ]]
then
  TRAIN_SETS=('MC-19-0,MC-19-1' 'MC-19-1,MC-19-2' 'MC-19-2,MC-19-0')
  TEST_SETS=('MC-19-2' 'MC-19-0' 'MC-19-1')
elif [[ $SET_ID == 21 ]]
then
  TRAIN_SETS=('MC-21-0' 'MC-21-1')
  TEST_SETS=('MC-21-1' 'MC-21-0')
elif [[ $SET_ID == 22 ]]
then
  TRAIN_SETS=('MC-22-0,MC-22-1' 'MC-22-1,MC-22-2' 'MC-22-2,MC-22-0')
  TEST_SETS=('MC-22-2' 'MC-22-0' 'MC-22-1')
elif [[ $SET_ID == 24 ]]
then
  TRAIN_SETS=('MC-24-0,MC-24-1' 'MC-24-1,MC-24-2' 'MC-24-2,MC-24-0')
  TEST_SETS=('MC-24-2' 'MC-24-0' 'MC-24-1')
elif [[ $SET_ID == 25 ]]
then
  TRAIN_SETS=('MC-25-0' 'MC-25-1')
  TEST_SETS=('MC-25-1' 'MC-25-0')
fi

if [ -n "$RESULTS_DIR" ]; then
  echo "RESULTS_DIR ALREADY SET: $RESULTS_DIR"
else
  RESULTS_DIR=./results_per_patient
  echo "RESULTS_DIR SET TO DEFAULT: $RESULTS_DIR"
fi

if [ -n "$N_EPOCHS" ]; then
  echo "N_EPOCHS ALREADY SET: $N_EPOCHS"
else
  N_EPOCHS=15
  echo "N_EPOCHS SET TO DEFAULT $N_EPOCHS"
fi

if [ -n "$MODEL_SAVE_DIR" ]; then
  echo "MODEL_SAVE_DIR ALREADY SET: $MODEL_SAVE_DIR"
else
  MODEL_SAVE_DIR=${RESULTS_DIR}/models
  echo "MODEL_SAVE_DIR SET TO DEFAULT: $MODEL_SAVE_DIR"
fi


mkdir -p $RESULTS_DIR
mkdir -p $MODEL_SAVE_DIR

NUM_BANDS=(1 2 4 8)
N_FILTERS=(16 32 64)


for (( i=0; i<${#TRAIN_SETS[*]}; i++ ));
do
  echo "Train: ${TRAIN_SETS[$i]}"
  echo "Test: ${TEST_SETS[$i]}"

  # Last line includes any extra arguments after the set ID
  python experiments.py \
    --result-dir=$RESULTS_DIR \
    --n-epochs=$N_EPOCHS \
    --save-model-path=$MODEL_SAVE_DIR \
    --train-sets=${TRAIN_SETS[$i]} \
    --test-sets=${TEST_SETS[$i]} \
    "${@:2}"

    #--track-sinc-params \
    #--sn-n-bands=$num_bands \
    #--n-cnn-filters=$n_filters \

    #--dataset=$DATASET \
    #--batchnorm \
    #--model-name=$MODEL_NAME \
    #--roll-channels \
    #--cv-sets=MC-24-0 \
    #--in-channel-dropout-rate=$IN_CHANNEL_DROPOUT \
done
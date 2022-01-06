#!/bin/bash

if [ $# -eq 0 ]
then
  SET_ID='MC-10'
  echo "Pass the set id - setting default: $SET_ID"
else
  SET_ID=$1
fi
echo "SET_ID: $SET_ID"

if [[ $SET_ID == 'NW-1' ]]
then
  #TRAIN_SETS=('NW-1-0,NW-1-1' 'NW-1-1,NW-1-2' 'NW-1-2,NW-1-0')
  #TEST_SETS=('NW-1-2' 'NW-1-0' 'NW-1-1')
  TRAIN_SETS=('NW-1-0' 'NW-1-1')
  TEST_SETS=('NW-1-1' 'NW-1-0')
elif [[ $SET_ID == 'NW-2' ]]
then
  TRAIN_SETS=('NW-2-0,NW-2-1,NW-2-2' 'NW-2-1,NW-2-2,NW-2-3' 'NW-2-2,NW-2-3,NW-2-0' 'NW-2-3,NW-2-0,NW-2-1')
  TEST_SETS=('NW-2-3' 'NW-2-0' 'NW-2-1', 'NW-2-2')
elif [[ $SET_ID == 'NW-3' ]]
then
  TRAIN_SETS=('NW-3-0' 'NW-3-1')
  TEST_SETS=('NW-3-1' 'NW-3-0')
elif [[ $SET_ID == 'NW-4' ]]
then
  TRAIN_SETS=('NW-4-0' 'NW-4-1')
  TEST_SETS=('NW-4-1' 'NW-4-0')
elif [[ $SET_ID == 'NW-5' ]]
then
  TRAIN_SETS=('NW-5-0,NW-5-1' 'NW-5-1,NW-5-2' 'NW-5-2,NW-5-0')
  TEST_SETS=('NW-5-2' 'NW-5-0' 'NW-5-1')
elif [[ $SET_ID == 'NW-6' ]]
then
  TRAIN_SETS=('NW-6-0' 'NW-6-1')
  TEST_SETS=('NW-6-1' 'NW-6-0')
elif [[ $SET_ID == 'MC-19' ]]
then
  TRAIN_SETS=('MC-19-0,MC-19-1' 'MC-19-1,MC-19-2' 'MC-19-2,MC-19-0')
  TEST_SETS=('MC-19-2' 'MC-19-0' 'MC-19-1')
elif [[ $SET_ID == 'MC-21' ]]
then
  TRAIN_SETS=('MC-21-0' 'MC-21-1')
  TEST_SETS=('MC-21-1' 'MC-21-0')
elif [[ $SET_ID == 'MC-22' ]]
then
  TRAIN_SETS=('MC-22-0,MC-22-1' 'MC-22-1,MC-22-2' 'MC-22-2,MC-22-0')
  TEST_SETS=('MC-22-2' 'MC-22-0' 'MC-22-1')
elif [[ $SET_ID == 'MC-24' ]]
then
  TRAIN_SETS=('MC-24-0,MC-24-1' 'MC-24-1,MC-24-2' 'MC-24-2,MC-24-0')
  TEST_SETS=('MC-24-2' 'MC-24-0' 'MC-24-1')
elif [[ $SET_ID == 'MC-25' ]]
then
  TRAIN_SETS=('MC-25-0' 'MC-25-1')
  TEST_SETS=('MC-25-1' 'MC-25-0')
elif [[ $SET_ID == 'MC-26' ]]
then
  TRAIN_SETS=('MC-26-0' 'MC-26-1')
  TEST_SETS=('MC-26-1' 'MC-26-0')
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

if [ -n "$EXEC_CMD" ]; then
  echo "EXEC_CMD ALREADY SET: $EXEC_CMD"
else
  EXEC_CMD="python experiments/standard.py"
  echo "EXEC_CMD SET TO DEFAULT: $EXEC_CMD"
fi

mkdir -p $RESULTS_DIR
mkdir -p $MODEL_SAVE_DIR

for (( i=0; i<${#TRAIN_SETS[*]}; i++ ));
do
  echo "Train: ${TRAIN_SETS[$i]}"
  echo "Test: ${TEST_SETS[$i]}"


  # Last line includes any extra arguments after the set ID
  $EXEC_CMD \
    --result-dir=$RESULTS_DIR \
    --n-epochs=$N_EPOCHS \
    --save-model-path=$MODEL_SAVE_DIR \
    --train-sets=${TRAIN_SETS[$i]} \
    --test-sets=${TEST_SETS[$i]} \
    "${@:2}"
done
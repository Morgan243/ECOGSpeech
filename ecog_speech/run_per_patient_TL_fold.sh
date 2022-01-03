# Quick script: Accepts a participant string (e.g. 'MC-21') and the
# remaining participants as --pre-train-sets to the ./run_pre_patient_fold.sh script
ALL_SETS=(MC-19 MC-21 MC-22 MC-24 MC-26)

if [ $# -eq 0 ]
then
  SET_ID='MC-19'
  echo "Pass the set id - setting default: $SET_ID"
else
  SET_ID=$1
fi
echo "SET_ID: $SET_ID"

# CREATE THE PRETRAINING DATA - EVERYBODY BUT THE SET ID
PRE_TRAIN_SETS=()
for value in "${ALL_SETS[@]}"
do
    [[ $value != $SET_ID ]] && PRE_TRAIN_SETS+=($value)
done
echo "PRE-TRAIN sets: ${PRE_TRAIN_SETS[*]}"

# Swap space seperate for comma separated
COMMA_SETS=$(IFS=,; printf '%s' "${PRE_TRAIN_SETS[*]}")
echo "comma-train sets: $COMMA_SETS"

time ./run_per_patient_fold.sh $SET_ID \
      --track-sinc-params \
      --pre-train-sets=$COMMA_SETS \
       "${@:2}"

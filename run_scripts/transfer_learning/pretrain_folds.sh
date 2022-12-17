
EXEC="time python -m ecog_speech.experiments.semi_supervised"
RESULT_PATH="../../results/cog2vec/pretrained/"
mkdir -p $RESULT_PATH/models

DEFAULT_CLI_ARGS="--batch_size=512 --learning_rate=0.0001 \
 --ppl_weight=1000 --quant_num_vars=70 --lr_adjust_patience=10 \
 --n_encoder_layers=5 --device='cuda' --feature_extractor_layers='[(128, 7, 3)] + [(128, 3, 2)] * 3 + [(64, 3, 1)]' \
 --flatten_sensors_to_samples=True --n_dl_workers=32 \
 --result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models --n_epochs=10"

#eval "$EXEC --train_sets=~UCSD-4 $DEFAULT_CLI_ARGS"
#eval "$EXEC  --train_sets=~UCSD-5 $DEFAULT_CLI_ARGS"

#eval "$EXEC  --train_sets=~UCSD-10 $DEFAULT_CLI_ARGS"
#eval "$EXEC  --train_sets=~UCSD-18 $DEFAULT_CLI_ARGS"

#eval "$EXEC  --train_sets=~UCSD-19 $DEFAULT_CLI_ARGS"
#eval "$EXEC  --train_sets=~UCSD-22 $DEFAULT_CLI_ARGS"

eval "$EXEC  --train_sets=~UCSD-28 $DEFAULT_CLI_ARGS"

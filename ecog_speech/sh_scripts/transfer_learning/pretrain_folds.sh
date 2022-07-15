EXEC="time python -m ecog_speech.experiments.semi_supervised"
RESULT_PATH="../../results/cog2vec/pretrained220714/"

mkdir -p $RESULT_PATH/models

#--model.quant_num_vars=100 --task.ppl_weight=1000 \
# --model.n_encoder_layers=7 \
DEFAULT_CLI_ARGS=" --dataset.flatten_sensors_to_samples=True \
--dataset.extra_output_keys='sensor_ras_coord_arr' --dataset.n_dl_workers=39 --dataset.n_dl_eval_workers=39 --dataset.batch_size=1500 \
--task.learning_rate=0.001 --task=semi_supervised \
--task.n_epochs=30 --task.device=cuda --dataset=hvs \
--model=cog2vec \
--dataset.pre_processing_pipeline=random_sample \
--result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models "

#DEFAULT_CLI_ARGS_OLD="--batch_size=512 --learning_rate=0.0001 \
# --ppl_weight=1000 --quant_num_vars=70 --lr_adjust_patience=10 \
# --n_encoder_layers=5 --device='cuda' --feature_extractor_layers='[(128, 7, 3)] + [(128, 3, 2)] * 3 + [(64, 3, 1)]' \
# --flatten_sensors_to_samples=True --n_dl_workers=32 \
# --result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models --n_epochs=10"

eval "$EXEC  --dataset.train_sets=~UCSD-19 $DEFAULT_CLI_ARGS"
eval "$EXEC  --dataset.train_sets=~UCSD-22 $DEFAULT_CLI_ARGS"
eval "$EXEC  --dataset.train_sets=~UCSD-28 $DEFAULT_CLI_ARGS"

eval "$EXEC --dataset.train_sets=~UCSD-4 $DEFAULT_CLI_ARGS"
eval "$EXEC  --dataset.train_sets=~UCSD-5 $DEFAULT_CLI_ARGS"
eval "$EXEC  --dataset.train_sets=~UCSD-10 $DEFAULT_CLI_ARGS"
eval "$EXEC  --dataset.train_sets=~UCSD-18 $DEFAULT_CLI_ARGS"


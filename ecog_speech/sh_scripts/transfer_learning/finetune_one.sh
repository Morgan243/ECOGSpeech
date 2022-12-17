EXEC="time python -m ecog_speech.experiments.transfer_learning"
#RESULT_PATH="../../results/cog2vec/pretrained220821/fine_tuned"
echo "RESULT PATH: $RESULT_PATH"

mkdir -p $RESULT_PATH/models

#--model.quant_num_vars=100 --task.ppl_weight=1000 \
# --model.n_encoder_layers=7 \
DEFAULT_CLI_ARGS="--task.learning_rate=0.001 \
 --n_dl_workers=8 \
--dataset=hvs --task.n_epochs=100 --extra_output_keys='sensor_ras_coord_arr' --batch_size=512 --batch_size_eval=768 \
--task.lr_adjust_patience=7 --task.early_stopping_patience=13 --batch_norm=True \
--dropout=0.75 --n_layers=2 --linear_hidden_n=16 \
--result_dir=$RESULT_PATH "



#DEFAULT_CLI_ARGS=" --dataset.flatten_sensors_to_samples=True \
#--dataset.extra_output_keys='sensor_ras_coord_arr' --dataset.n_dl_workers=39 --dataset.n_dl_eval_workers=39 \
#--dataset.batch_size=1024 --dataset.batch_size_eval=2048 \
#--task.learning_rate=0.001 --task=semi_supervised \
#--task.lr_adjust_patience=7 --task.early_stopping_patience=13 \
#--task.n_epochs=100 --task.device=cuda --dataset=hvs \
#--model=cog2vec \
#--dataset.pre_processing_pipeline=random_sample \
#--result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models "

eval "$EXEC $DEFAULT_CLI_ARGS --task=region_detection $@"
eval "$EXEC $DEFAULT_CLI_ARGS --task=word_detection $@"
eval "$EXEC $DEFAULT_CLI_ARGS --task=speech_detection $@"

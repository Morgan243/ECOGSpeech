
./pretrain_one.sh --dataset.train_sets=~UCSD-4 "$@"
sleep 10
./pretrain_one.sh --dataset.train_sets=~UCSD-5 "$@"
sleep 10
./pretrain_one.sh --dataset.train_sets=~UCSD-10 "$@"
sleep 10
./pretrain_one.sh --dataset.train_sets=~UCSD-18 "$@"
sleep 10
./pretrain_one.sh --dataset.train_sets=~UCSD-19 "$@"
sleep 10
./pretrain_one.sh --dataset.train_sets=~UCSD-22 "$@"
sleep 10
./pretrain_one.sh --dataset.train_sets=~UCSD-28 "$@"

#DEFAULT_CLI_ARGS_OLD="--batch_size=512 --learning_rate=0.0001 \
# --ppl_weight=1000 --quant_num_vars=70 --lr_adjust_patience=10 \
# --n_encoder_layers=5 --device='cuda' --feature_extractor_layers='[(128, 7, 3)] + [(128, 3, 2)] * 3 + [(64, 3, 1)]' \
# --flatten_sensors_to_samples=True --n_dl_workers=32 \
# --result_dir=$RESULT_PATH --save_model_path=$RESULT_PATH/models --n_epochs=10"

#eval "$EXEC  --dataset.train_sets=~UCSD-19 $DEFAULT_CLI_ARGS"
#sleep 10
#eval "$EXEC  --dataset.train_sets=~UCSD-22 $DEFAULT_CLI_ARGS"
#sleep 10
# Below skipped - CV loss goes to zero ?!?!
#eval "$EXEC  --dataset.train_sets=~UCSD-28 $DEFAULT_CLI_ARGS"
#sleep 10
#eval "$EXEC --dataset.train_sets=~UCSD-4 $DEFAULT_CLI_ARGS"
#sleep 10
#eval "$EXEC  --dataset.train_sets=~UCSD-5 $DEFAULT_CLI_ARGS"
#sleep 10
#eval "$EXEC  --dataset.train_sets=~UCSD-10 $DEFAULT_CLI_ARGS"
#sleep 10
#eval "$EXEC  --dataset.train_sets=~UCSD-18 $DEFAULT_CLI_ARGS"


# Standard
TODO - docs
# Semi-supervised
Trains semi-supervised Wave2vec2-like (Cog2Vec) model
```bash
python -m ecog_speech.experiments.semi_supervised --batch_size=128 --train_sets=UCSD-28 --task.learning_rate=0.0001 --task.ppl_weight=100 --model.quant_num_vars=30 --task.lr_adjust_patience=10 --model.n_encoder_layers=3 --task.device='cpu' --model.feature_extractor_layers='[(128, 7, 3)] + [(128, 3, 2)] * 3 + [(128, 3, 1)]'
```
# Transfer Learning
From pre-trained results, choose a downstream task to fine tune on
```bash
python -m ecog_speech.experiments.transfer_learning --result_file=./results/cog2vec/pretrained2207/20220704_2125_aff657b0-1c6f-4482-ab98-c9b42f03eac4.json --model_base_path=./results/cog2vec/pretrained2207/models/ --dataset=hvs --extra_output_keys='sensor_ras_coord_arr' --pre_processing_pipeline=region_classification --task=region_detection
```


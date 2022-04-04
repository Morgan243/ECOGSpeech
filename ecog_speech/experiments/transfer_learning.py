import uuid
import time
from datetime import datetime
from os.path import join as pjoin
import json
import torch
from ecog_speech import utils
from ecog_speech.experiments.standard import train_and_test_model
from ecog_speech.models import base
from typing import List, Optional

from ecog_speech.experiments.standard import make_model, make_datasets_and_loaders, default_option_kwargs
from ecog_speech.experiments import standard, semi_supervised
from ecog_speech.experiments import base as bxp
from dataclasses import dataclass
import json

logger = utils.get_logger(__name__)

@dataclass
class TransferLearningOptions(bxp.DNNModelOptions, bxp.MultiSensorOptions):
    model_name: Optional[str] = None
    dataset: Optional[str] = None
    train_sets: Optional[str] = None

    pretrained_result_input_path: Optional[str] = None
    pretrained_result_model_base_path: Optional[str] = None

    pre_train_sets: Optional[str] = None
    pre_cv_sets: Optional[str] = None
    pre_test_sets: Optional[str] = None


def run_sincnet(options: TransferLearningOptions):
    ##################
    def make_sub_results(stage, _trainer, _dataset_map, _outputs_map, _performance_map):
        return dict(stage=stage,
                    batch_losses=_trainer.epoch_res_map,
                    train_selected_columns=_dataset_map['train'].selected_columns,
                    best_model_epoch=_trainer.best_model_epoch,
                    **utils.make_classification_reports({k + "_clf_report": d
                                                         for k, d in _outputs_map.items()}),
                    **{f"{part_name}_{metric_name}": metric_value
                       for part_name, perf_d in _performance_map.items()
                       for metric_name, metric_value in perf_d.items()})

    #####
    # Load pre-training data and initialize a fresh model from it
    logger.info("Loading pre-training data")
    pre_dataset_map, pre_dl_map, pre_eval_dl_map = make_datasets_and_loaders(options,
                                                                             train_sets_str=options.pre_train_sets,
                                                                             cv_sets_str=options.pre_cv_sets,
                                                                             test_sets_str=options.pre_test_sets)

    logger.info("Initializing new model")
    model, model_kws = make_model(options, pre_dataset_map['train'])
    logger.info("Pre-training model")
    pre_trainer, pre_outputs_map, pre_performance_map = train_and_test_model(model, pre_dl_map,
                                                                             pre_eval_dl_map, options)
    pre_results_d = make_sub_results('pretraining', pre_trainer, pre_dataset_map, pre_outputs_map, pre_performance_map)
    # columns used throughout are determined by the pretraining's selected from its training's valid sensors
    # Future TODO - may want to consider other aspects or something more generic?
    selected_columns = pre_dataset_map['train'].selected_columns

    batch_cb_history = getattr(pre_trainer, 'batch_cb_history', None)#.get('band_params', None)
    #pre_band_params = getattr(pre_trainer, 'batch_cb_history', dict()).get('band_params', None)
    if batch_cb_history is not None:
        models_batch_results = model.format_results(batch_cb_history)
        pre_results_d.update(models_batch_results)
    #pre_results_d['low_hz_frame'], pre_results_d['high_hz_frame'] = (parse_band_params(pre_band_params, model.fs)
    #                                                                 if pre_band_params is not None else (None, None))

    ### Fine-tuning
    logger.info("Loading fine-tuning data")
    dataset_map, dl_map, eval_dl_map = make_datasets_and_loaders(options,
                                                                 train_sensor_columns=selected_columns,
                                                                 train_sets_str=options.train_sets,
                                                                 cv_sets_str=options.cv_sets,
                                                                 test_sets_str=options.test_sets)
    logger.info("Fine-tuning model")
    trainer, outputs_map, performance_map = train_and_test_model(model, dl_map, eval_dl_map, options,
                                                                 # Don't overwrite the weights that were pre-trained
                                                                 weights_init_f=None)

    results_d = make_sub_results('finetuning', trainer, dataset_map, outputs_map, performance_map)
    logger.info("Fine-tuning complete")

    band_params = getattr(trainer, 'batch_cb_history', dict()).get('band_params', None)
    if band_params is not None:
        models_batch_results = model.format_results(band_params)
        results_d.update(models_batch_results)

    #results_d['low_hz_frame'], results_d['high_hz_frame'] = (parse_band_params(band_params, model.fs)
    #                                                         if band_params is not None else (None, None))

    uid = str(uuid.uuid4())
    results_d['uid'] = pre_results_d['uid'] = uid
    t = int(time.time())
    name = "%d_%s_TL" % (t, uid)
    file_name = name + '.json'
    res_dict = dict(name=name,
                    file_name=file_name,
                    datetime=str(datetime.now()), uid=uid,
                    train_selected_columns=selected_columns,
                    num_trainable_params=utils.number_of_model_params(model),
                    num_params=utils.number_of_model_params(model, trainable_only=False),
                    model_kws=model_kws,

                    pretraining_results=pre_results_d,
                    finetuning_results=results_d,

                    **vars(options))

    if options.save_model_path is not None:
        import os
        p = options.save_model_path
        if os.path.isdir(p):
            p = os.path.join(p, uid + '.torch')
        logger.info("Saving model to " + p)
        torch.save(model.cpu().state_dict(), p)
        res_dict['save_model_path'] = p

    if options.result_dir is not None:
        path = pjoin(options.result_dir, file_name)
        logger.info(path)
        res_dict['path'] = path
        with open(path, 'w') as f:
            json.dump(res_dict, f)


def run(options: TransferLearningOptions):
    # Pretrained model already prepared, parse from its results output
    if options.pretrained_result_input_path is not None:
        from ecog_speech.result_parsing import load_model_from_results

        result_path = options.pretrained_result_input_path
        model_base_path = options.pretrained_result_model_base_path

        print(f"Loading pretrained model from results in {result_path} (base path = {model_base_path})")
        with open(result_path, 'r') as f:
            result_json = json.load(f)

        pretrained_model = load_model_from_results(result_json, base_model_path=model_base_path)
    # Need to pre-train the model now
    else:
        raise NotImplementedError()
        # Need to pretrain
        pass

    fine_tune_model = pretrained_model.create_fine_tuning_model()
    dataset_map, dl_map, eval_dl_map = semi_supervised.make_datasets_and_loaders(options)

    ft_trainer = base.Trainer(model_map=dict(model=fine_tune_model), opt_map=dict(),
                              train_data_gen=dl_map['train'],
                              cv_data_gen=dl_map.get('cv'),
                              input_key='signal_arr',
                              learning_rate=options.learning_rate,
                              early_stopping_patience=options.early_stopping_patience,
                              device=options.device,
                              )
    ft_results = ft_trainer.train(options.n_epochs)

    fine_tune_model.eval()
    outputs_map = ft_trainer.generate_outputs(**eval_dl_map)
    #eval_res_map = {k: ft_trainer.eval_on(_dl).to_dict(orient='list') for k, _dl in eval_dl_map.items()}
    clf_str_map = utils.make_classification_reports(outputs_map)
    performance_map = {part_name: utils.performance(outputs_d['actuals'], outputs_d['preds'] > 0.5)
                       for part_name, outputs_d in outputs_map.items()}


all_model_hyperparam_names = TransferLearningOptions.get_all_model_hyperparam_names()


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(description="ASPEN+MHRG Transfer Learning experiments")
    parser.add_arguments(TransferLearningOptions, dest='transfer_learning')
    args = parser.parse_args()
    main_options: TransferLearningOptions = args.transfer_learning
    #main_options.pretrained_result_input_path = '../../results/cog2vec/1649000864_7a68aaf5-e41f-4f4e-bb56-0b0ca0a2a4fb_TL.json'
    #main_options.pretrained_result_model_base_path = '../../results/cog2vec/models/'
    #main_options.dataset = 'hvs'
    #main_options.train_sets = 'UCSD-22'
    #main_options.flatten_sensors_to_samples = True
    #main_options.pre_processing_pipeline = 'audio_gate'

    run(main_options)

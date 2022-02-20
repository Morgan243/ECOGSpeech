import uuid
import time
from datetime import datetime
from os.path import join as pjoin
import json
import torch
from ecog_speech import utils
from ecog_speech.experiments.standard import train_and_test_model
from ecog_speech.models import base

from ecog_speech.experiments.standard import make_model, make_datasets_and_loaders, default_option_kwargs

logger = utils.get_logger(__name__)


def run(options):
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

    ##################

    def parse_band_params(_band_params, fs):
        lowhz_df_map, highhz_df_map, centerhz_df_map = base.BaseMultiSincNN.parse_band_parameter_training_hist(
            _band_params,
            fs=fs)
        if model.per_channel_filter:
            _low_hz = {k: lowhz_df.to_json() for k, lowhz_df in lowhz_df_map.items()}
            _high_hz = {k: highhz_df.to_json() for k, highhz_df in highhz_df_map.items()}
        else:
            _low_hz = lowhz_df_map[0].to_json()
            _high_hz = highhz_df_map[0].to_json()
        return _low_hz, _high_hz

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

    pre_band_params = getattr(pre_trainer, 'batch_cb_history', dict()).get('band_params', None)
    pre_results_d['low_hz_frame'], pre_results_d['high_hz_frame'] = (parse_band_params(pre_band_params, model.fs)
                                                                     if pre_band_params is not None else (None, None))

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
    results_d['low_hz_frame'], results_d['high_hz_frame'] = (parse_band_params(band_params, model.fs)
                                                             if band_params is not None else (None, None))

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


tl_options = [
    dict(dest='--pre-train-sets', default=None, type=str),
    dict(dest='--pre-cv-sets', default=None, type=str),
    dict(dest='--pre-test-sets', default=None, type=str),
]
tl_option_kwargs = default_option_kwargs + tl_options


all_model_hyperparam_names = [d['dest'].replace('--', '').replace('-', '_')
                              for d in tl_options
                              if d['dest'] not in ('--train-sets', '--cv-sets', '--test-sets')]

if __name__ == """__main__""":
    parser = utils.build_argparse(tl_option_kwargs,
                                  description="ASPEN+MHRG Transfer Learning experiments")
    m_options = parser.parse_args()
    run(m_options)

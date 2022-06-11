import uuid
import time
from datetime import datetime
from os.path import join as pjoin
import json
import torch
from ecog_speech import datasets, utils
from ecog_speech.models import base, sinc_ieeg
from typing import Optional, Type
from ecog_speech.experiments import base as bxp
from ecog_speech.models import base as bmp

logger = utils.get_logger(__name__)


def make_model(options: Type[bmp.DNNModelOptions] = None, nww=None, model_name=None, model_kws=None, print_details=True):
    """
    Helper method - Given command-line options and a NorthwesterWords derived dataset, build the model
    specified in the options.
    """
    assert not (nww is None and model_kws is None)
    base_kws = dict()
    if options is not None:
        base_kws.update(dict(
            #window_size=int(nww.sample_ixer.window_size.total_seconds() * nww.fs_signal),
            # time/samples is the last dimension
            window_size=int(nww.get_feature_shape()[-1]),
            dropout=options.dropout,
            dropout2d=options.dropout_2d,
            batch_norm=options.batchnorm,
            dense_width=options.dense_width,
            activation_cls=options.activation_class,
            print_details=print_details
        ))

    model_name = options.model_name if model_name is None else model_name

    if model_name == 'base-sn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             n_bands=options.sn_n_bands,
                             n_cnn_filters=options.n_cnn_filters,
                             sn_padding=options.sn_padding,
                             sn_kernel_size=options.sn_kernel_size,
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             fs=nww.fs_signal,
                             cog_attn=options.cog_attn,
                             **base_kws)
        model = base.BaseMultiSincNN(**model_kws)
    elif model_name == 'tnorm-base-sn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             n_bands=options.sn_n_bands,
                             n_cnn_filters=options.n_cnn_filters,
                             sn_padding=options.sn_padding,
                             sn_kernel_size=options.sn_kernel_size,
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             fs=nww.fs_signal,
                             cog_attn=options.cog_attn,
                             band_spacing=options.sn_band_spacing,
                             **base_kws)
        model = sinc_ieeg.TimeNormBaseMultiSincNN(**model_kws)
    elif 'tnorm-base-sn-v' in model_name:
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             n_bands=options.sn_n_bands,
                             n_cnn_filters=options.n_cnn_filters,
                             sn_padding=options.sn_padding,
                             sn_kernel_size=options.sn_kernel_size,
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             fs=nww.fs_signal,
                             cog_attn=options.cog_attn,
                             band_spacing=options.sn_band_spacing,
                             **base_kws)
        model_cls = sinc_ieeg.get_model_cls_from_options_str(model_name)
        model = model_cls(**model_kws)
    elif model_name == 'base-cnn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             n_cnn_filters=options.n_cnn_filters,
                             # band_spacing=options.sn_band_spacing,
                             **base_kws)
        model = base.BaseCNN(**model_kws)
    else:
        msg = f"Unknown model name {model_name}"
        raise ValueError(msg)

    return model, model_kws


#def train_and_test_model(model, dl_map, eval_dl_map, options, Trainer_CLS=base.Trainer,
#                         **trainer_kws):
#    reg_f = None
#    if options.bw_reg_weight > 0:
#        logger.warning(f"!!!! Using BW regularizeer W={options.bw_reg_weight} !!!!")
#        reg_f = lambda m: model.bandwidth_regularizer(m, w=options.bw_reg_weight)
#
#    batch_cb = None
#    sn_params_tracked = False
#    if options.track_sinc_params and 'sn' in options.model_name:
#        sn_params_tracked = True
#        batch_cb = dict(band_params=model.get_band_params)
#    elif options.track_sinc_params:
#        logger.warning("--track-sinc-params was set, but not using an SN model - ignoring!")
#
#    logger.debug(Trainer_CLS)
#    logger.debug(base.__file__)
#    trainer = Trainer_CLS(model_map=dict(model=model),
#                          opt_map=dict(),
#                          train_data_gen=dl_map['train'],
#                          cv_data_gen=dl_map.get('cv'),
#                          model_regularizer=reg_f,
#                          learning_rate=options.learning_rate,
#                          early_stopping_patience=options.early_stopping_patience,
#                          device=options.device,
#                          **trainer_kws)
#
#    logger.info("Training")
#    losses = trainer.train(options.n_epochs,
#                           batch_callbacks=batch_cb,
#                           batch_cb_delta=5)
#    logger.info("Reloading best model state")
#    model.load_state_dict(trainer.get_best_state())
#
#    #####
#    # Produce predictions and score them
#    model.eval()
#    outputs_map = trainer.generate_outputs(**eval_dl_map)
#    clf_str_map = utils.make_classification_reports(outputs_map)
#
#    performance_map = {part_name: utils.performance(outputs_d['actuals'], outputs_d['preds'] > 0.5)
#                       for part_name, outputs_d in outputs_map.items()}
#
#    return trainer, outputs_map, performance_map
#
#    #####
#    # Prep a results structure for saving - everything must be json serializable (no array objects)
#

from dataclasses import dataclass, field
from ecog_speech.experiments import base as bxp
from ecog_speech.models import base as bmp
from simple_parsing import ArgumentParser, choice, subgroups


@dataclass
class SupervisedSpeechDetectionTask(bxp.TaskOptions):
    task_name: str = 'supervised_speech_detection'
    dataset = datasets.NorthwesternWordsDatasetOptions

@dataclass
class StandardExperiment(bxp.Experiment):
    model: bmp.ModelOptions = subgroups(
        {"sinc_ieeg": sinc_ieeg.SincIEEGOptions,
         "base_cnn": bmp.BaseCNNModelOptions},
        default=sinc_ieeg.SincIEEGOptions()
    )

    task: SupervisedSpeechDetectionTask = SupervisedSpeechDetectionTask
    #task: bxp.TaskOptions = bxp.TaskOptions('supervised_speech_detection',
    #                                        dataset=datasets.NorthwesternWordsDatasetOptions
                                            #dataset=datasets.DatasetOptions('nww', train_sets='MC-21-0')
                                            #)

    @staticmethod
    def train_and_test_model_orig(model, dl_map, eval_dl_map, options, Trainer_CLS=base.Trainer,
                                  **trainer_kws):
        reg_f = None
        if options.bw_reg_weight > 0:
            logger.warning(f"!!!! Using BW regularizeer W={options.bw_reg_weight} !!!!")
            reg_f = lambda m: model.bandwidth_regularizer(m, w=options.bw_reg_weight)

        batch_cb = None
        sn_params_tracked = False
        if options.track_sinc_params and 'sn' in options.model_name:
            sn_params_tracked = True
            batch_cb = dict(band_params=model.get_band_params)
        elif options.track_sinc_params:
            logger.warning("--track-sinc-params was set, but not using an SN model - ignoring!")

        logger.debug(Trainer_CLS)
        logger.debug(base.__file__)
        trainer = Trainer_CLS(model_map=dict(model=model),
                              opt_map=dict(),
                              train_data_gen=dl_map['train'],
                              cv_data_gen=dl_map.get('cv'),
                              model_regularizer=reg_f,
                              learning_rate=options.learning_rate,
                              early_stopping_patience=options.early_stopping_patience,
                              device=options.device,
                              **trainer_kws)

        logger.info("Training")
        losses = trainer.train(options.n_epochs,
                               batch_callbacks=batch_cb,
                               batch_cb_delta=5)
        logger.info("Reloading best model state")
        model.load_state_dict(trainer.get_best_state())

        #####
        # Produce predictions and score them
        model.eval()
        outputs_map = trainer.generate_outputs(**eval_dl_map)
        clf_str_map = utils.make_classification_reports(outputs_map)

        performance_map = {part_name: utils.performance(outputs_d['actuals'], outputs_d['preds'] > 0.5)
                           for part_name, outputs_d in outputs_map.items()}

        return trainer, outputs_map, performance_map

    @classmethod
    def train_and_test_model(cls, model_opts: bmp.ModelOptions, task_opts: bxp.TaskOptions):
        dataset_map, dl_map, eval_dl_map = task_opts.dataset.make_datasets_and_loaders()
        model, model_kws = model_opts.make_model(dataset_map['train'])
        reg_func = model_opts.make_model_regularizer_function(model)

        batch_cb = None
        sn_params_tracked = getattr(model_opts, 'track_sinc_params', False)
        if sn_params_tracked:
            batch_cb = dict(band_params=model.get_band_params)

        trainer = base.Trainer(dict(model=model), opt_map=dict(),
                               train_data_gen=dl_map['train'],
                               cv_data_gen=dl_map.get('cv'),
                               model_regularizer=reg_func,
                               learning_rate=task_opts.learning_rate,
                               early_stopping_patience=task_opts.early_stopping_patience,
                               device=task_opts.device)

        logger.info("Training")
        losses = trainer.train(task_opts.n_epochs,
                               batch_callbacks=batch_cb,
                               batch_cb_delta=5)
        model.load_state_dict(trainer.get_best_state())

        #####
        # Produce predictions and score them
        model.eval()
        outputs_map = trainer.generate_outputs(**eval_dl_map)
        clf_str_map = utils.make_classification_reports(outputs_map)

        performance_map = {part_name: utils.performance(outputs_d['actuals'], outputs_d['preds'] > 0.5)
                           for part_name, outputs_d in outputs_map.items()}

        return dataset_map, model_kws, trainer, losses, outputs_map, performance_map

    def run(self):
        """
        Run an experiment using options used with argument parser (CLI).

        Parameters
        ----------
        options: opbject
        If using as a library, use the utils.build_default_options and pass the default_option_kwargs variable in this
        module as the default_option_kwargs.

        Returns
        -------
        Trainer, outputs_map
            A two-tuple of (1) Instance of Trainer and (2) dictionary of partition results
        """

        dataset_map, model_kws, trainer, losses, outputs_map, performance_map = self.train_and_test_model(self.model, self.task)

        model = trainer.model_map['model']

        res_dict = self.create_result_dictionary(
            batch_losses=losses,
            train_selected_columns=dataset_map['train'].selected_columns,
            best_model_epoch=trainer.best_model_epoch,
            num_trainable_params=utils.number_of_model_params(model),
            num_params=utils.number_of_model_params(model, trainable_only=False),
            model_kws=model_kws,
            #**{'train_' + k: v for k, v in train_perf_map.items()},
            #**{'cv_' + k: v for k, v in cv_perf_map.items()},
            #**test_perf_map,
            **{'train_' + k: v for k, v in performance_map['train'].items()},
            **{'cv_' + k: v for k, v in performance_map['cv'].items()},
            **performance_map['test'],
            **vars(self))

        uid = res_dict['uid']
        name = res_dict['name']

        if self.result_output.save_model_path is not None:
            import os
            p = self.result_output.save_model_path
            if os.path.isdir(p):
                p = os.path.join(p, uid + '.torch')
            logger.info("Saving model to " + p)
            torch.save(model.cpu().state_dict(), p)
            res_dict['save_model_path'] = p

        # if options.track_sinc_params:
        if getattr(self.model, 'track_sinc_params', False):
            lowhz_df_map, highhz_df_map, centerhz_df_map = base.BaseMultiSincNN.parse_band_parameter_training_hist(
                trainer.batch_cb_history['band_params'],
                fs=model.fs)
            if model.per_channel_filter:
                res_dict['low_hz_frame'] = {k: lowhz_df.to_json() for k, lowhz_df in lowhz_df_map.items()}
                res_dict['high_hz_frame'] = {k: highhz_df.to_json() for k, highhz_df in highhz_df_map.items()}
            else:
                res_dict['low_hz_frame'] = lowhz_df_map[0].to_json()
                res_dict['high_hz_frame'] = highhz_df_map[0].to_json()

        if self.result_output.result_dir is not None:
            path = pjoin(self.result_output.result_dir, name)
            logger.info(path)
            res_dict['path'] = path
            with open(path, 'w') as f:
                json.dump(res_dict, f)

        return trainer, outputs_map


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser(description="'standard' supervised classification experiments")
    parser.add_arguments(StandardExperiment, dest='experiment')
    args = parser.parse_args()
    experiment: StandardExperiment = args.experiment
    experiment.run()

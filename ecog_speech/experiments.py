import uuid
import time
from datetime import datetime
from os.path import join as pjoin
import json
import torch
from ecog_speech import datasets, utils
from ecog_speech.models import base


def make_model(options=None, nww=None, model_name=None, model_kws=None):
    """
    Helper method - Given command-line options and a NorthwesterWords derived dataset, build the model
    specified in the options.
    """
    assert not (nww is None and model_kws is None)
    if options is not None:
        base_kws = dict(
            window_size=int(nww.sample_ixer.window_size.total_seconds() * nww.fs_signal),
            dropout=options.dropout,
            dropout2d=options.dropout_2d,
            batch_norm=options.batchnorm,
            dense_width=options.dense_width,
            activation_cls=options.activation_class,
        )

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
        model = base.TimeNormBaseMultiSincNN(**model_kws)
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
        # Grab the right model - should have done a dict lookup or something...
        if 'v2' in model_name:
            model = base.TimeNormBaseMultiSincNN_v2(**model_kws)
        elif 'v3' in model_name:
            model = base.TimeNormBaseMultiSincNN_v3(**model_kws)
        #elif 'v4' in model_name:
        #    model = base.TimeNormBaseMultiSincNN_v4(**model_kws)
        elif 'v5' in model_name:
            model = base.TimeNormBaseMultiSincNN_v5(**model_kws)
        elif 'v6' in model_name:
            model = base.TimeNormBaseMultiSincNN_v6(**model_kws)
        elif 'v7' in model_name:
            model = base.TimeNormBaseMultiSincNN_v7(**model_kws)
        elif 'v8' in model_name:
            model = base.TimeNormBaseMultiSincNN_v8(**model_kws)
        elif 'v9' in model_name:
            model = base.TimeNormBaseMultiSincNN_v9(**model_kws)
        elif 'v10' in model_name:
            model = base.TimeNormBaseMultiSincNN_v10(**model_kws)
        elif 'v11' in model_name:
            model = base.TimeNormBaseMultiSincNN_v11(**model_kws)

    elif model_name == 'base-cnn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             n_cnn_filters=options.n_cnn_filters,
                             #band_spacing=options.sn_band_spacing,
                             **base_kws)
        model = base.BaseCNN(**model_kws)
    else:
        msg = f"Unknown model name {model_name}"
        raise ValueError(msg)

    return model, model_kws


def make_datasets_and_loaders(options, dataset_cls=None, train_data_kws=None, cv_data_kws=None, test_data_kws=None,
                              num_dl_workers=8) -> tuple:
    """
    Helper method to create instances of dataset_cls as specified in the command-line options and
    additional keyword args.
    Parameters
    ----------
    options: object
        Options object build using the utils module
    dataset_cls: Derived class of BaseDataset (default=None)
        E.g. NorthwesterWords
    train_data_kws: dict (default=None)
        keyword args to train version of the dataset
    cv_data_kws: dict (default=None)
        keyword args to cv version of the dataset
    test_data_kws: dict (default=None)
        keyword args to test version of the dataset
    num_dl_workers: int (default=8)
        Number of workers in each dataloader. Can be I/O bound, so sometimes okay to over-provision

    Returns
    -------
    dataset_map, dataloader_map, eval_dataloader_map
        three-tuple of (1) map to original dataset (2) map to the constructed dataloaders and
        (3) Similar to two, but not shuffled and larger batch size (for evaluation)
    """
    from torchvision import transforms
    if dataset_cls is None:
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(options.dataset)

    train_p_tuples = dataset_cls.make_tuples_from_sets_str(options.train_sets)
    cv_p_tuples = dataset_cls.make_tuples_from_sets_str(options.cv_sets)
    test_p_tuples = dataset_cls.make_tuples_from_sets_str(options.test_sets)

    if train_data_kws is None:
        train_data_kws = dict(patient_tuples=train_p_tuples, pre_processing_pipeline=options.pre_processing_pipeline)
    if cv_data_kws is None:
        cv_data_kws = dict(patient_tuples=cv_p_tuples, pre_processing_pipeline=options.pre_processing_pipeline)
    if test_data_kws is None:
        test_data_kws = dict(patient_tuples=test_p_tuples, pre_processing_pipeline=options.pre_processing_pipeline)

    dl_kws = dict(num_workers=num_dl_workers, batch_size=options.batch_size,
                  shuffle=False, random_sample=True)
    eval_dl_kws = dict(num_workers=num_dl_workers, batch_size=512,
                       shuffle=False, random_sample=False)


    dataset_map = dict()
    print("Using dataset class: %s" % str(dataset_cls))
    train_nww = dataset_cls(power_q=options.power_q,
                            sensor_columns='valid',
                            **train_data_kws)
    if options.roll_channels and options.shuffle_channels:
        raise ValueError("--roll-channels and --shuffle-channels are mutually exclusive")
    elif options.roll_channels:
        print("-->Rolling channels transform<--")
        train_nww.transform = transforms.Compose([
            datasets.RollDimension(roll_dim=0, min_roll=0,
                                   max_roll=train_nww.sensor_count - 1)
        ])
    elif options.shuffle_channels:
        print("-->Shuffle channels transform<--")
        train_nww.transform = transforms.Compose([
            datasets.ShuffleDimension()
        ])

    if options.random_labels:
        print("-->Randomizing target labels<--")
        train_nww.target_transform = transforms.Compose([
            datasets.RandomIntLike(low=0, high=2)
        ])

    dataset_map['train'] = train_nww

    if cv_data_kws['patient_tuples'] is not None:
        dataset_map['cv'] = dataset_cls(power_q=options.power_q,
                                        sensor_columns=train_nww.selected_columns,
                                         **cv_data_kws)
    else:
        from sklearn.model_selection import train_test_split
        train_ixs, cv_ixes = train_test_split(range(len(train_nww)))
        cv_nww = dataset_cls(data_from=train_nww).select(cv_ixes)
        train_nww.select(train_ixs)
        dataset_map.update(dict(train=train_nww,
                                cv=cv_nww))

    dataset_map['test'] = dataset_cls(power_q=options.power_q,
                                      sensor_columns=train_nww.selected_columns,
                                        **test_data_kws)

    #dataset_map = dict(train=train_nww, cv=cv_nww, test=test_nww)

    dataloader_map = {k: v.to_dataloader(**dl_kws)
                      for k, v in dataset_map.items()}
    eval_dataloader_map = {k: v.to_dataloader(**eval_dl_kws)
                                for k, v in dataset_map.items()}

    return dataset_map, dataloader_map, eval_dataloader_map


def run(options):
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
    dataset_map, dl_map, eval_dl_map = make_datasets_and_loaders(options)
    model, model_kws = make_model(options, dataset_map['train'])
    print("Building trainer")
    reg_f = None
    if options.bw_reg_weight > 0:
        print(f"!!!! Using BW regularizeer W={options.bw_reg_weight} !!!!")
        reg_f = lambda m: model.bandwidth_regularizer(m, w=options.bw_reg_weight)

    batch_cb = None
    sn_params_tracked = False
    if options.track_sinc_params and 'sn' in options.model_name:
        sn_params_tracked = True
        batch_cb = dict(band_params=model.get_band_params)
    elif options.track_sinc_params:
        print("--track-sinc-params was set, but not using an SN model - ignoring!")

    trainer = base.Trainer(dict(model=model), opt_map=dict(),
                           train_data_gen=dl_map['train'],
                           cv_data_gen=dl_map.get('cv'),
                           model_regularizer=reg_f,
                           learning_rate=options.learning_rate,
                           early_stopping_patience=options.early_stopping_patience,
                           device=options.device)

    print("Training")
    losses = trainer.train(options.n_epochs,
                           batch_callbacks=batch_cb,
                           batch_cb_delta=5)
    model.load_state_dict(trainer.get_best_state())

    #####
    # Produce predictions and score them
    model.eval()
    outputs_map = trainer.generate_outputs(**eval_dl_map)
    clf_str_map = utils.make_classification_reports(outputs_map)

    train_perf_map = utils.performance(outputs_map['train']['actuals'],
                                       outputs_map['train']['preds'] > 0.5)
    cv_perf_map = utils.performance(outputs_map['cv']['actuals'],
                                   outputs_map['cv']['preds'] > 0.5)
    test_perf_map = utils.performance(outputs_map['test']['actuals'],
                                      outputs_map['test']['preds'] > 0.5)
    #####
    # Prep a results structure for saving - everything must be json serializable (no array objects)
    uid = str(uuid.uuid4())
    t = int(time.time())
    name = "%d_%s_TL.json" % (t, uid)
    res_dict = dict(#path=path,
                    name=name,
                    datetime=str(datetime.now()), uid=uid,
                    #batch_losses=list(losses),
                    batch_losses=losses,
                    train_selected_columns=dataset_map['train'].selected_columns,
                    best_model_epoch=trainer.best_model_epoch,
                    num_trainable_params=utils.number_of_model_params(model),
                    num_params=utils.number_of_model_params(model, trainable_only=False),
                    model_kws=model_kws,
                    clf_reports=clf_str_map,
                    **{'train_'+k: v for k, v in train_perf_map.items()},
                    **{'cv_'+k: v for k, v in cv_perf_map.items()},
                    **test_perf_map,
                    #evaluation_perf_map=perf_maps,
                    #**pretrain_res,
                    #**perf_map,
        **vars(options))
    if options.save_model_path is not None:
        import os
        p = options.save_model_path
        if os.path.isdir(p):
            p = os.path.join(p, uid + '.torch')
        print("Saving model to " + p)
        torch.save(model.cpu().state_dict(), p)
        res_dict['save_model_path'] = p

    #if options.track_sinc_params:
    if sn_params_tracked:
        lowhz_df_map, highhz_df_map, centerhz_df_map = base.BaseMultiSincNN.parse_band_parameter_training_hist(
            trainer.batch_cb_history['band_params'],
            fs=model.fs)
        if model.per_channel_filter:
            res_dict['low_hz_frame'] = {k: lowhz_df.to_json() for k, lowhz_df in lowhz_df_map.items()}
            res_dict['high_hz_frame'] = {k: highhz_df.to_json() for k, highhz_df in highhz_df_map.items()}
        else:
            res_dict['low_hz_frame'] = lowhz_df_map[0].to_json()
            res_dict['high_hz_frame'] = highhz_df_map[0].to_json()

    if options.result_dir is not None:
        path = pjoin(options.result_dir, name)
        print(path)
        res_dict['path'] = path
        with open(path, 'w') as f:
            json.dump(res_dict, f)

    return trainer, outputs_map


default_model_hyperparam_option_kwargs = [
    dict(dest='--model-name', default='base-sn', type=str),
    dict(dest='--dataset', default='nww', type=str),
    dict(dest='--train-sets', default='MC-19-0,MC-19-1', type=str),
    # dict(dest='--cv-sets', default='19-2', type=str),
    dict(dest='--cv-sets', default=None, type=str),
    dict(dest='--test-sets', default='MC-19-2', type=str),
    dict(dest='--random-labels', default=False, action="store_true"),
    dict(dest='--pre-processing-pipeline', default='default', type=str),

    dict(dest='--learning-rate', default=0.001, type=float),
    dict(dest='--dense-width', default=None, type=int),
    dict(dest='--sn-n-bands', default=1, type=int),
    dict(dest='--sn-kernel-size', default=31, type=int),
    dict(dest='--sn-padding', default=15, type=int),
    dict(dest='--sn-band-spacing', default='linear', type=str),
    dict(dest='--n-cnn-filters', default=None, type=int),
    dict(dest='--activation-class', default='PReLU', type=str),
    dict(dest='--dropout', default=0., type=float),
    dict(dest='--dropout-2d', default=False, action="store_true"),
    dict(dest='--in-channel-dropout-rate', default=0., type=float),
    dict(dest='--batchnorm', default=False, action="store_true"),
    dict(dest='--roll-channels', default=False, action="store_true"),
    dict(dest='--shuffle-channels', default=False, action="store_true"),
    dict(dest='--cog-attn', default=False, action="store_true"),
    dict(dest='--power-q', default=0.7, type=float),
    dict(dest='--n-epochs', default=100, type=int),
    dict(dest='--early-stopping-patience', default=None, type=int),
    dict(dest='--batch-size', default=256, type=int),
    dict(dest='--bw-reg-weight', default=0.0, type=float),
]

all_model_hyperparam_names = [d['dest'].replace('--', '').replace('-', '_')
                           for d in default_model_hyperparam_option_kwargs
                            if d['dest'] not in ('--train-sets', '--cv-sets', '--test-sets')]

default_option_kwargs = default_model_hyperparam_option_kwargs + [
    dict(dest='--track-sinc-params', default=False, action="store_true"),
    dict(dest='--device', default='cuda:0'),
    dict(dest='--save-model-path', default=None),
    dict(dest='--tag', default=None),
    dict(dest='--result-dir', default=None),
]

if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs,
                                  description="ASPEN+MHRG Experiments v1")
    m_options = parser.parse_args()
    results = run(m_options)

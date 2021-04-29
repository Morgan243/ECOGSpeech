import uuid
import time
from datetime import datetime
from os.path import join as pjoin
import json
from ecog_speech import datasets
import pandas as pd
import numpy as np
import matplotlib
import torch
from ecog_speech import datasets, feature_processing, utils
from tqdm.auto import tqdm
from ecog_speech.models import base
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score


def make_model(options, nww):
    base_kws = dict(
        window_size=int(nww.sample_ixer.window_size.total_seconds() * nww.fs_signal),
        dropout=options.dropout,
        dropout2d=options.dropout_2d,
        batch_norm=options.batchnorm,
        dense_width=options.dense_width,
    )

    if options.model_name == 'base-sn':
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
    elif options.model_name == 'tnorm-base-sn':
        model_kws = dict(in_channels=len(nww.selected_columns),
                         n_bands=options.sn_n_bands,
                         n_cnn_filters=options.n_cnn_filters,
                         sn_padding=options.sn_padding,
                         sn_kernel_size=options.sn_kernel_size,
                         in_channel_dropout_rate=options.in_channel_dropout_rate,
                         fs=nww.fs_signal,
                         cog_attn=options.cog_attn,
                         **base_kws)
        model = base.TimeNormBaseMultiSincNN(**model_kws)
    elif options.model_name == 'base-cnn':
        model_kws = dict(in_channels=len(nww.selected_columns),
                         in_channel_dropout_rate=options.in_channel_dropout_rate,
                         n_cnn_filters=options.n_cnn_filters,
                         **base_kws)
        model = base.BaseCNN(**model_kws)
    else:
        msg = f"Unknown model name {options.model_name}"
        raise ValueError(msg)

    return model, model_kws


def make_datasets_and_loaders(options, dataset_cls=None, train_data_kws=None, cv_data_kws=None, test_data_kws=None,
                              num_dl_workers=8):
    from torchvision import transforms
    if dataset_cls is None:
        dataset_cls = datasets.BaseDataset.get_dataset_by_name(options.dataset)

    train_p_tuples = dataset_cls.make_tuples_from_sets_str(options.train_sets)
    cv_p_tuples = dataset_cls.make_tuples_from_sets_str(options.cv_sets)
    test_p_tuples = dataset_cls.make_tuples_from_sets_str(options.test_sets)

    if train_data_kws is None:
        train_data_kws = dict(patient_tuples=train_p_tuples)
    if cv_data_kws is None:
        cv_data_kws = dict(patient_tuples=cv_p_tuples)
    if test_data_kws is None:
        test_data_kws = dict(patient_tuples=test_p_tuples)

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
                                   max_roll=len(train_nww.default_sensor_columns) - 1)
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


def run_simple(options):
    dataset_map, dl_map, eval_dl_map = make_datasets_and_loaders(options)
    model, model_kws = make_model(options, dataset_map['train'])
    print("Building trainer")
    reg_f = None
    if options.bw_reg_weight > 0:
        print(f"!!!! Using BW regularizeer W={options.bw_reg_weight} !!!!")
        reg_f = lambda m: model.bandwidth_regularizer(m, w=options.bw_reg_weight)

    batch_cb = None
    if options.track_sinc_params:
        batch_cb = dict(band_params=model.get_band_params)

    trainer = base.Trainer(dict(model=model), opt_map = dict(),
                           train_data_gen=dl_map['train'],
                           cv_data_gen=dl_map.get('cv'),
                           model_regularizer=reg_f,
                           learning_rate=options.learning_rate,
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
                    best_model_epoch=trainer.best_model_epoch,
                    num_trainable_params=utils.number_of_model_params(model),
                    num_params=utils.number_of_model_params(model, trainable_only=False),
                    model_kws=model_kws,
                    clf_reports=clf_str_map,
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

    if options.track_sinc_params:
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

def run_kfold(options):
    #nww-kfold-19
    pass


def run(options):
    if 'kfold' in options.dataset:
        return run_kfold(options)
    else:
        return run_simple(options)


def example_run(options):

    ######
    # First some preprocessing that's a little hacky since
    # we load up the whole dataset first to determine where
    # the labels are, then remake datasets from the partitions
    #####


    # Loads the dataset and uses some feature processing
    # to find areas where speech is happening and label it
    nww = datasets.NorthwesternWords(power_q=options.power_q,
                                     patient_tuples=(('Mayo Clinic', 19, 1, 1),
                                                        #('Mayo Clinic', 19, 1, 2),
                                                        #('Mayo Clinic', 19, 1, 3)
                                                        ),
                                     #ecog_window_n=75
                                     )
    from torchvision import transforms
    transform = None
    if options.roll_channels:
        transform = transforms.Compose([
            datasets.RollDimension(roll_dim=0, max_roll=len(nww.default_sensor_columns) - 1)
        ])

    # Word index series is the same length as the speech data
    # and values idetify speaking regions (negative values are
    # the preceding silence), so take the max to get the number
    # of silence/speech pairs
    #all_ixes = range(int(nww.word_index.max() + 1))
    all_ixes = range(nww.flat_keys.shape[0])

    # partition the speech and silence pairs
    #train_ixes, cv_ixes = train_test_split(all_ixes, train_size=.7)
    #cv_ixes, test_ixes = train_test_split(cv_ixes, train_size=.5)


    ######
    # Now we are ready to create our datasets for training and testing
    #####
    train_nww = datasets.NorthwesternWords(#selected_word_indices=train_ixes,
        transform=transform,
                                              data_from=nww)
    #cv_nww = data_loader.NorthwesternWords(selected_word_indices=cv_ixes,
    #                                       data_from=nww)
    cv_nww = datasets.NorthwesternWords(patient_tuples=(('Mayo Clinic', 19, 1, 2),),
                                        power_q=options.power_q)
    #test_nww = data_loader.NorthwesternWords(selected_word_indices=test_ixes,
    #                                         data_from=nww)
    test_nww = datasets.NorthwesternWords(patient_tuples=(('Mayo Clinic', 19, 1, 3),),
                                          power_q=options.power_q)

    # Get pytorch dataloaders
    dl_kws = dict(num_workers=4, batch_size=256,
                  shuffle=False, random_sample=True)
    print("Building data loader")
    train_dl = train_nww.to_dataloader(**dl_kws)
    cv_dl = cv_nww.to_dataloader(**dl_kws)
    #test_dl = test_nww.to_dataloader(**dl_kws)

    print("Building model")
    # starter test model for detecting speech from brain waves
    # Learn band extraction at the top
    model, model_kws = make_model(options, nww)

    print(model)
    ####
    dl_eval_kws = dict(num_workers=4, batch_size=256,
                       shuffle=False, random_sample=False)
    eval_dset_map = dict(train_dl=train_nww.to_dataloader(**dl_eval_kws),
                         cv_dl=cv_nww.to_dataloader(**dl_eval_kws),
                         test_dl=test_nww.to_dataloader(**dl_eval_kws))

    print("Building trainer")
    trainer = base.Trainer(model=model, train_data_gen=train_dl,
                           cv_data_gen=cv_dl)
    print("Training")
    losses = trainer.train(options.n_epochs)
    model.load_state_dict(trainer.get_best_state())
    trainer.model.eval()

    outputs_map = trainer.generate_outputs(**eval_dset_map)
    utils.make_classification_reports(outputs_map)
    test_perf_map = utils.performance(outputs_map['test_dl']['actuals'],
                                      outputs_map['test_dl']['preds'] > 0.5)

    if options.save_model_path is not None:
        print("Saving model to " + options.save_model_path)
        torch.save(trainer.model.state_dict(), options.save_model_path)


###
    uid = str(uuid.uuid4())
    t = int(time.time())
    name = "%d_%s_TL.json" % (t, uid)
    res_dict = dict(#path=path,
                    name=name,
                    datetime=str(datetime.now()), uid=uid,
                    batch_losses=list(losses),
                    num_trainable_params=utils.number_of_model_params(model),
                    num_params=utils.number_of_model_params(model, trainable_only=False),
                    **test_perf_map,
                    #evaluation_perf_map=perf_maps,
                    #**pretrain_res,
                    #**perf_map,
        **vars(options))

    if options.result_dir is not None:
        path = pjoin(options.result_dir, name)
        print(path)
        res_dict['path'] = path
        with open(path, 'w') as f:
            json.dump(res_dict, f)

    return trainer, outputs_map



default_option_kwargs = [
    dict(dest='--model-name', default='base-sn', type=str),
    dict(dest='--dataset', default='nww', type=str),
    dict(dest='--train-sets', default='MC-19-0,MC-19-1', type=str),
    #dict(dest='--cv-sets', default='19-2', type=str),
    dict(dest='--cv-sets', default=None, type=str),
    dict(dest='--test-sets', default='MC-19-2', type=str),
    dict(dest='--random-labels', default=False, action="store_true"),

    dict(dest='--learning-rate', default=0.001, type=float),
    dict(dest='--dense-width', default=None, type=int),
    dict(dest='--sn-n-bands', default=1, type=int),
    dict(dest='--sn-kernel-size', default=31, type=int),
    dict(dest='--sn-padding', default=15, type=int),
    dict(dest='--n-cnn-filters', default=None, type=int),
    dict(dest='--dropout', default=0., type=float),
    dict(dest='--dropout-2d', default=False, action="store_true"),
    dict(dest='--in-channel-dropout-rate', default=0., type=float),
    dict(dest='--batchnorm', default=False, action="store_true"),
    dict(dest='--roll-channels', default=False, action="store_true"),
    dict(dest='--shuffle-channels', default=False, action="store_true"),
    dict(dest='--cog-attn', default=False, action="store_true"),
    dict(dest='--bw-reg-weight', default=0.0, type=float),
    dict(dest='--track-sinc-params', default=False, action="store_true"),
    #dict(dest='--batch-callback-delta', default=5, type=int),

    dict(dest='--power-q', default=0.7, type=float),

    dict(dest='--n-epochs', default=100, type=int),
    dict(dest='--batch-size', default=256, type=int),
    dict(dest='--device', default='cuda:0'),
    dict(dest='--save-model-path', default=None),
    dict(dest='--tag', default=None),
    dict(dest='--result-dir', default=None),
]

# TODO: Config, CLI, etc

if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs,
                                  description="ASPEN+MHRG Experiments v1")
    m_options = parser.parse_args()
    results = run(m_options)

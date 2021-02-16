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

def process_outputs(output_map):
    for dname, o_map in output_map.items():
        print("-"*10 + str(dname) + "-"*10)
        print(classification_report(o_map['actuals'], (o_map['preds'] > 0.5)))

def make_model(options, nww):
    base_kws = dict(
        #window_size=nww.ecog_window_size,
        window_size=int(nww.sample_ixer.window_size.total_seconds() * nww.fs_signal),
        dropout=options.dropout,
        dropout2d=options.dropout_2d,
        batch_norm=options.batchnorm,
        dense_width=options.dense_width,
    )

    if options.model_name == 'base-sn':
        model = base.BaseMultiSincNN(len(nww.sensor_columns),
                                     n_bands=options.sn_n_bands,
                                     n_cnn_filters=options.n_cnn_filters,
                                     sn_padding=options.sn_padding,
                                     sn_kernel_size=options.sn_kernel_size,
                                     fs=nww.fs_signal,
                                     **base_kws)
    elif options.model_name == 'base-cnn':
        model = base.BaseCNN(len(nww.sensor_columns), **base_kws)
    else:
        msg = f"Unknown model name {options.model_name}"
        raise ValueError(msg)

    return model

def make_datasets_and_loaders(options):
    from torchvision import transforms
    dl_kws = dict(num_workers=4, batch_size=options.batch_size,
                  shuffle=False, random_sample=True)
    eval_dl_kws = dict(num_workers=4, batch_size=512,
                  shuffle=False, random_sample=False)

    if options.dataset == 'nww':
        train_nww = datasets.NorthwesternWords(power_q=options.power_q,
                                               patient_tuples=(('Mayo Clinic', 19, 1, 1),
                                                            # ('Mayo Clinic', 19, 1, 2),
                                                            # ('Mayo Clinic', 19, 1, 3)
                                                            ),
                                               # ecog_window_n=75
                                               )
        if options.roll_channels:
            train_nww.transform = transforms.Compose([
                datasets.RollDimension(roll_dim=0,
                                       max_roll=len(train_nww.sensor_columns) - 1)
            ])
        cv_nww = datasets.NorthwesternWords(patient_tuples=(('Mayo Clinic', 19, 1, 2),),
                                            power_q=options.power_q)
        test_nww = datasets.NorthwesternWords(patient_tuples=(('Mayo Clinic', 19, 1, 3),),
                                              power_q=options.power_q)

        dataset_map = dict(train=train_nww, cv=cv_nww, test=test_nww)
        #dataloader_map = {k: v.to_dataloader(**dl_kws)
        #                  for k, v in dataset_map.items()}
        #return dataset_map, dataloader_map

    elif options.dataset == 'chang-nww':
        train_nww = datasets.ChangNWW(power_q=options.power_q,
                                      patient_tuples=(
                                             ('Mayo Clinic', 19, 1, 2),
                                             #('Mayo Clinic', 21, 1, 2),
                                             #('Mayo Clinic', 22, 1, 2),
                                         ),
                                      )
        if options.roll_channels:
            train_nww.transform = transforms.Compose([
                datasets.RollDimension(roll_dim=0,
                                       max_roll=len(train_nww.sensor_columns) - 1)
            ])

        cv_nww = datasets.ChangNWW(power_q=options.power_q,
                                   patient_tuples=(
                                             ('Mayo Clinic', 24, 1, 2),
                                         ))

        test_nww = datasets.ChangNWW(power_q=options.power_q,
                                     patient_tuples=(
                                          ('Mayo Clinic', 25, 1, 2),
                                      ))

        dataset_map = dict(train=train_nww, cv=cv_nww, test=test_nww)
    else:
        msg = f"Unknown dataset: '{options.dataset}'"
        raise ValueError(msg)

    dataloader_map = {k: v.to_dataloader(**dl_kws)
                      for k, v in dataset_map.items()}
    eval_dataloader_map = {k: v.to_dataloader(**eval_dl_kws)
                                for k, v in dataset_map.items()}
    return dataset_map, dataloader_map, eval_dataloader_map


def run(options):
    dataset_map, dl_map, eval_dl_map = make_datasets_and_loaders(options)
    model = make_model(options, dataset_map['train'])
    print("Building trainer")
    trainer = base.Trainer(dict(model=model), opt_map = dict(),
                    train_data_gen = dl_map['train'],
                    cv_data_gen = dl_map['cv'])
    #trainer = base.Trainer(model=model, train_data_gen=dl_map['train'],
    #                       cv_data_gen=dl_map['cv'])


    print("Training")
    losses = trainer.train(options.n_epochs)
    model.load_state_dict(trainer.get_best_state())
    #trainer.model_map['model'].eval()
    model.eval()

    outputs_map = trainer.generate_outputs(**eval_dl_map)
    process_outputs(outputs_map)
    test_perf_map = utils.performance(outputs_map['test']['actuals'],
                                      outputs_map['test']['preds'] > 0.5)

    if options.save_model_path is not None:
        print("Saving model to " + options.save_model_path)
        torch.save(trainer.model.state_dict(), options.save_model_path)

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
            datasets.RollDimension(roll_dim=0, max_roll=len(nww.sensor_columns) - 1)
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
    model = make_model(options, nww)

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
    process_outputs(outputs_map)
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

    dict(dest='--dense-width', default=None, type=int),
    dict(dest='--sn-n-bands', default=1, type=int),
    dict(dest='--sn-kernel-size', default=31, type=int),
    dict(dest='--sn-padding', default=0, type=int),
    dict(dest='--n-cnn-filters', default=None, type=int),
    dict(dest='--dropout', default=0., type=float),
    dict(dest='--dropout-2d', default=False, action="store_true"),
    dict(dest='--batchnorm', default=False, action="store_true"),
    dict(dest='--roll-channels', default=False, action="store_true"),

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
    parser = utils.build_argparse(default_option_kwargs, description="ASPEN+MHRG Experiments v1")
    options = parser.parse_args()
    results = run(options)
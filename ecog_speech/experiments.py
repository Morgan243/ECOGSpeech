from ecog_speech import data_loader
import pandas as pd
import numpy as np
import matplotlib
import torch
from ecog_speech import data_loader, feature_processing, utils
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
    #model = base.BaseMultiSincNN()
    model = base.BaseMultiSincNN(64, window_size=nww.max_ecog_window_size,
                                 sn_kernel_size=options.sn_kernel_size,
                                 dropout=options.dropout,
                                 dropout2d=options.dropout_2d,
                                 batch_norm=options.batchnorm,
                                 n_bands=options.sn_n_bands,
                                 sn_padding=options.sn_padding,
                             fs=nww.ecog_sample_rate)
    return model

def run(options):

    ######
    # First some preprocessing that's a little hacky since
    # we load up the whole dataset first to determine where
    # the labels are, then remake datasets from the partitions
    #####


    # Loads the dataset and uses some feature processing
    # to find areas where speech is happening and label it
    nww = data_loader.NorthwesternWords(power_q=options.power_q,
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
            data_loader.RollDimension(roll_dim=0, max_roll=len(nww.sensor_columns) - 1)
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
    train_nww = data_loader.NorthwesternWords(#selected_word_indices=train_ixes,
        transform=transform,
                                              data_from=nww)
    #cv_nww = data_loader.NorthwesternWords(selected_word_indices=cv_ixes,
    #                                       data_from=nww)
    cv_nww = data_loader.NorthwesternWords(patient_tuples=(('Mayo Clinic', 19, 1, 2),),
                                             power_q=options.power_q)
    #test_nww = data_loader.NorthwesternWords(selected_word_indices=test_ixes,
    #                                         data_from=nww)
    test_nww = data_loader.NorthwesternWords(patient_tuples=(('Mayo Clinic', 19, 1, 3),),
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
    dl_eval_kws = dict(num_workers=4, batch_size=256, shuffle=False, random_sample=False)
    eval_dset_map = dict(train_dl=train_nww.to_dataloader(**dl_eval_kws),
                         cv_dl=cv_nww.to_dataloader(**dl_eval_kws),
                         test_dl=test_nww.to_dataloader(**dl_eval_kws))

    print("Building trainer")
    trainer = base.Trainer(model=model, train_data_gen=train_dl, cv_data_gen=cv_dl)
    print("Training")
    trainer.train(options.n_epochs)
    model.load_state_dict(trainer.get_best_state())
    trainer.model.eval()

    outputs_map = trainer.generate_outputs(**eval_dset_map)
    process_outputs(outputs_map)

    if options.save_model_path is not None:
        print("Saving model to " + options.save_model_path)
        torch.save(trainer.model.state_dict(), options.save_model_path)

    return trainer, outputs_map



default_option_kwargs = [
    dict(dest='--dense-width', default=64, type=int),
    dict(dest='--sn-n-bands', default=2, type=int),
    dict(dest='--sn-kernel-size', default=31, type=int),
    dict(dest='--sn-padding', default=0, type=int),
    dict(dest='--dropout', default=0., type=float),
    dict(dest='--dropout-2d', default=False, action="store_true"),
    dict(dest='--batchnorm', default=False, action="store_true"),
    dict(dest='--roll-channels', default=False, action="store_true"),

    dict(dest='--power-q', default=0.5, type=float),

    dict(dest='--n-epochs', default=100, type=int),
    dict(dest='--device', default='cuda:0'),
    dict(dest='--save-model-path', default=None),
    dict(dest='--result-dir', default=None),
]

# TODO: Config, CLI, etc

if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs, description="ASPEN+MHRG Experiments v1")
    options = parser.parse_args()
    results = run(options)
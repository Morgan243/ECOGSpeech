from ecog_speech import data_loader
import pandas as pd
import numpy as np
import matplotlib
import torch
from ecog_speech import data_loader, feature_processing
from tqdm.auto import tqdm
from ecog_speech.models import base
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def generate_outputs(m, **dl_map):
    output_map = dict()
    for dname, dset in dl_map.items():
        preds_l, actuals_l = list(), list()
        for _x in tqdm(dset):
            preds_l.append(m(_x['ecog_arr']))
            actuals_l.append(_x['text_arr'])

        output_map[dname] = dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
                                 actuals=torch.cat(actuals_l).detach().cpu().int().numpy())
    return output_map

from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
def process_outputs(output_map):
    for dname, o_map in output_map.items():
        print("-"*10 + str(dname) + "-"*10)
        print(classification_report(o_map['actuals'], (o_map['preds'] > 0.5)))


def run(options):

    ######
    # First some preprocessing that's a little hacky since
    # we load up the whole dataset first to determine where
    # the labels are, then remake datasets from the partitions
    #####

    # Loads the dataset and uses some feature processing
    # to find areas where speech is happening and label it
    nww = data_loader.NorthwesternWords()

    # Word index series is the same length as the speech data
    # and values idetify speaking regions (negative values are
    # the preceding silence), so take the max to get the number
    # of silence/speech pairs
    all_ixes = range(int(nww.word_index.max() + 1))

    # partition the speech and silence pairs
    train_ixes, cv_ixes = train_test_split(all_ixes, train_size=.7)
    cv_ixes, test_ixes = train_test_split(cv_ixes, train_size=.5)

    # Add in the silence as negative label of the word id
    train_ixes += [-v for v in train_ixes]
    cv_ixes += [-v for v in cv_ixes]
    test_ixes += [-v for v in test_ixes]


    ######
    # Now we are ready to create our datasets for training and testing
    #####
    train_nww = data_loader.NorthwesternWords(selected_word_indices=train_ixes)
    cv_nww = data_loader.NorthwesternWords(selected_word_indices=cv_ixes)
    test_nww = data_loader.NorthwesternWords(selected_word_indices=test_ixes)

    # Get pytorch dataloaders
    dl_kws = dict(num_workers=4, batch_size=128, shuffle=True, random_sample=False)
    train_dl = train_nww.to_dataloader(**dl_kws)
    cv_dl = cv_nww.to_dataloader(**dl_kws)
    #test_dl = test_nww.to_dataloader(**dl_kws)

    # starter test model for detecting speech from brain waves
    # Learn band extraction at the top
    m = torch.nn.Sequential(
        base.Unsqueeze(2),
        # Pctile(),
        base.MultiChannelSincNN(2, 64, kernel_size=31, fs=train_nww.ecog_sample_rate, per_channel_filter=True),
        torch.nn.Dropout2d(),
        torch.nn.Conv2d(64, 64, kernel_size=(1, 5), stride=(1, 5), dilation=(1, 1), groups=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.PReLU(),

        # torch.nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5)),
        torch.nn.Dropout2d(),
        torch.nn.Conv2d(64, 64, kernel_size=(1, 5), stride=(1, 5)),
        torch.nn.BatchNorm2d(64),
        torch.nn.PReLU(),

        torch.nn.Dropout2d(),
        torch.nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 3)),
        torch.nn.BatchNorm2d(64),
        torch.nn.PReLU(),

        torch.nn.Dropout2d(),
        torch.nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2)),
        torch.nn.BatchNorm2d(128),
        torch.nn.PReLU(),

        base.Flatten(),
        torch.nn.Dropout(),

        torch.nn.Linear(768, 1), torch.nn.Sigmoid()
    )

    ####
    eval_dset_map = dict(train_dl=train_nww.to_dataloader(random_sample=False, shuffle=False),
                         cv_dl=cv_nww.to_dataloader(random_sample=False, shuffle=False),
                         test_dl=test_nww.to_dataloader(random_sample=False, shuffle=False))

    trainer = base.Trainer(model=m, train_data_gen=train_dl, cv_data_gen=cv_dl)
    trainer.train(20)
    m.load_state_dict(trainer.get_best_state())
    trainer.model.eval()

    outputs_map = generate_outputs(m, **eval_dset_map)
    process_outputs(outputs_map)



default_option_kwargs = [
    dict(dest='--dense-depth', default=1, type=int),
    dict(dest='--dense-width', default=64, type=int),
]

# TODO: Config, CLI, etc

if __name__ == """__main__""":
    run(None)
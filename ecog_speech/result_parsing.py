import pandas as pd
import numpy as np
import matplotlib
from glob import glob
import os
import json
from ecog_speech import datasets, feature_processing, experiments, utils
from ecog_speech.models import base
from tqdm.auto import tqdm
import torch

def plot_model_preds(preds_s, data_map, sample_index_map):
    plt_stim_s = data_map['stim']
    #t_indices_s = {wrd_id: (nww.sample_index_maps[data_k].get(wrd_id), nww.sample_index_maps[data_k].get(-wrd_id))
    t_indices_s = {wrd_id: (sample_index_map.get(wrd_id), sample_index_map.get(-wrd_id))
                   for wrd_id in plt_stim_s.unique() if wrd_id != 0}

    plt_dfs, neg_plt_dfs = list(), list()
    for wrd_id, (wrd_ix, sil_ix) in t_indices_s.items():
        # _cols = [f'neg_{wrd_id}', f'{wrd_id}']
        s = pd.Series(0, index=data_map['stim'].index, name=wrd_id)
        s.loc[wrd_ix[0].min():wrd_ix[-1].max()] = 1
        plt_dfs.append(s)

        s = pd.Series(0, index=data_map['stim'].index, name=-wrd_id)
        s.loc[sil_ix[0].min():sil_ix[-1].max()] = 1
        neg_plt_dfs.append(s)

    plt_label_df = pd.concat(plt_dfs, axis=1)

    ix_min, ix_max = '0s', '20s'
    td_ix_min, td_ix_max = pd.Timedelta(ix_min), pd.Timedelta(ix_max)
    td_win_size = td_ix_max - td_ix_min

    n_rows = int((data_map['audio'].index.max() / td_win_size) + 0.5)
    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, figsize=(35, 5 * n_rows))

    ax = None
    for i, ax in enumerate(axs):
        plt_ix_min = td_ix_min + (td_win_size * i)
        plt_ix_max = td_ix_max + (td_win_size * i)
        plt_audio = data_map['audio']

        plt_audio_mask = (plt_audio.index > plt_ix_min) & (plt_audio.index < plt_ix_max)
        plt_label_mask = (plt_label_df.index > plt_ix_min) & (plt_label_df.index < plt_ix_max)

        plt_audio = plt_audio.loc[plt_audio_mask].reindex(plt_label_df.loc[plt_label_mask].index,
                                                          method='ffill')  # .fillna(method='ffill')

        ax = (plt_audio / plt_audio.quantile(0.999)).clip(-1, 1).plot(color='grey', alpha=0.8, lw=0.75, ax=ax,
                                                                      sharex=False, label='audio')
        ax.set_ylabel(f"Row {i}: {plt_ix_min.total_seconds()} seconds", fontsize=18)
        ax2 = ax.twinx()
        ax = plt_label_df.any(axis=1).astype(int).loc[plt_ix_min:plt_ix_max].plot(  # color='tab:blue',
            lw=7, ls='--', grid=True, alpha=0.7, legend=False,  # figsize=(35, 6),
            title='Labeled word regions and speech prediction' if i == 0 else '',
            label='labeled speaking',
            sharex=False,
            ax=ax2)
        plt_pred_s = preds_s.loc[plt_ix_min:plt_ix_max].rolling(50).mean()
        plt_pred_s.plot(ax=ax2, color='tab:green', lw=5, label='predicted speaking proba')
        (plt_pred_s > 0.5).astype(int).plot(ax=ax2, color='orange', ls='--', label='predicted speaking gt 0.5')
        if i == 0:
            ax.legend(fontsize=17)
            ax2.legend(fontsize=17)
    fig.tight_layout()

    return fig, ax

def eval_nww_model(model, nww, win_step=1):
    model_preds = dict()
    print(f"Running {len(nww.data_maps)} eval data map(s): {', '.join(map(str, nww.data_maps.keys()))}")
    for ptuple, data_map in nww.data_maps.items():
        ecog_torch_arr = torch.from_numpy(data_map['ecog'].values)
        win_size = nww.ecog_window_size
        # TODO: seems like there should be a better way to do this
        all_ecog_dl = torch.utils.data.DataLoader([ecog_torch_arr[_ix:_ix + win_size].T
                                                   for _ix in range(0, ecog_torch_arr.shape[0] - win_size, win_step)],
                                                  batch_size=1024, num_workers=6)
        with torch.no_grad():
            all_ecog_out = [model(x) for x in tqdm(all_ecog_dl)]


        all_ecog_pred_s = pd.Series([_v.item() for v in all_ecog_out for _v in v],
                                    index=data_map['ecog'].iloc[
                                        range(win_size, ecog_torch_arr.shape[0], win_step)].index,
                                    name='pred_proba')
        #model_preds[ptuple] = all_ecog_out
        model_preds[ptuple] = all_ecog_pred_s

    return model_preds

dataset_evaluator_map = dict(
    nww=eval_nww_model,
)

def run(options):
    if options.result_file is None:
        raise ValueError("Must provide result-file option")

    ###############
    ### Path handling
    result_base_path, result_filename = os.path.split(options.result_file)
    result_id = result_filename.split('.')[0]
    # Load results to get the file name of the model
    results = json.load(open(options.result_file))

    model_filename = os.path.split(results['save_model_path'])[-1]
    base_model_path = options.base_model_path
    if base_model_path is None:
        base_model_path = os.path.join(result_base_path, 'models')
        print("Base model path not give - assuming path '%s'" % base_model_path)


    base_output_path =  result_id if options.base_output_path is None else options.base_output_path
    from pathlib import Path
    print(f"Creating results dir {base_output_path} if it doesn't already exist")
    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    #import os
    #if not os.path.exists(base_output_path):
    #    os.makedirs(base_output_path)

    ###############
    ### Processing
    model_kws = results['model_kws']
    loss_df = pd.DataFrame(results['batch_losses'])
    ax = loss_df.plot(figsize=(6, 5), grid=True, lw=3)
    ax.figure.savefig(os.path.join(base_model_path, "training_losses.pdf"))

    lowhz_df = pd.read_json(results['low_hz_frame']).sort_index()
    highhz_df = pd.read_json(results['high_hz_frame']).sort_index()
    centerhz_df = (highhz_df + lowhz_df) / 2.

    ax = None
    for c in lowhz_df.columns:
        ax = centerhz_df[c].plot(figsize=(15, 6), lw=3, ax=ax)
        ax.fill_between(centerhz_df.index,
                        lowhz_df[c],
                        highhz_df[c],
                        alpha=0.5)
    ax.grid(True)
    ax.figure.savefig(os.path.join(base_model_path, "band_param_training_plot.pdf"))
    # ax.legend(False)

    if options.eval_sets is not None:
        model_path = os.path.join(base_model_path, model_filename)
        print("Loading model located at: " + str(model_path))

        #eval_set_str = dict(test="test_sets", cv="cv_sets", train="train_sets").get(options.eval_sets,
        #                                                                            options.eval_sets)
        eval_set_str = {k: results[k + "_sets"] for k in ["test", "cv", "train"]}.get(options.eval_sets,
                                                                                      options.eval_sets)

        data_k_l = experiments.make_tuples_from_sets_str(eval_set_str)


        model = base.BaseMultiSincNN(**model_kws)

        # with open('../ecog_speech/test_results/t_cog_attn_model_2.torch', 'rb') as f:
        with open(model_path, 'rb') as f:
            model_state = torch.load(f)
            # mut_state = pickle.load(f)

        model.load_state_dict(model_state)

        #dset_cls = dataset_evaluator_map.get(results['dataset'])
        #dset = dset_cls()
        nww = datasets.NorthwesternWords(patient_tuples=data_k_l)
        preds_map = eval_nww_model(model, nww, options.eval_win_step_size)
        for ptuple, data_map in nww.data_maps.items():
            print("Plotting " + str(ptuple))
            ax, fig = plot_model_preds(preds_s=preds_map[ptuple], data_map=data_map,
                                       sample_index_map=nww.sample_index_maps[ptuple])
            fig.savefig(os.path.join(base_output_path, "prediction_plot_for_%s.pdf" % str(ptuple)))


default_option_kwargs = [
    dict(dest="--result-file", default=None, type=str),
    dict(dest="--base-model-path", default=None, type=str),
    dict(dest='--eval-sets', default=None, type=str,
         help="Dataset to run the loaded model against - use train/cv/test for the data used to build the model"
              "or specify the dataset name (e.g. MC-19-0. If unset, model will not be evaluated on data"),

    dict(dest="--eval-win-step-size", default=1, type=int),
    dict(dest="--base-output-path", default=None, type=str),
]
if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs,
                                  description="ASPEN+MHRG Result Parsing")
    options = parser.parse_args()
    results = run(options)

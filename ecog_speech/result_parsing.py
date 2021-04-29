import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
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
    fig, axs = plt.subplots(nrows=n_rows, figsize=(35, 5 * n_rows))

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

#dataset_evaluator_map = dict(
#    nww=eval_nww_model,
#)
#
def plot_training(loss_df, title=None, ax=None):
    ax = loss_df.plot(figsize=(6, 5), grid=True, lw=3, ax=ax)
    ax.set_ylabel('Loss Value', fontsize=13)
    ax.set_xlabel('Epoch', fontsize=13)
    if title is not None:
        ax.set_title(title, fontsize=15)
    #plt.clf()
    return ax


def plot_sensor_band_training(lowhz_df, centerhz_df, highhz_df,
                              title=None, ax=None):
    for c in lowhz_df.columns:
        ax = centerhz_df[c].plot(figsize=(15, 6), lw=3, ax=ax)
        ax.fill_between(centerhz_df.index,
                        lowhz_df[c],
                        highhz_df[c],
                        alpha=0.5)
    if title is not None:
        ax.set_title(title, fontsize=15)
    ax.set_ylabel('Hz', fontsize=13)
    # TODO: Can we map this to actual batches?
    ax.set_xlabel('Batch Sample Index', fontsize=13)
    ax.grid(True)
    #plt.clf()
    return ax


def multi_plot_training(loss_df, lowhz_df, centerhz_df,  highhz_df,
                        title=None,
                        axs=None):
    fig = None
    if axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    axs = axs.reshape(-1)

    ax_i = 0
    loss_ax = plot_training(loss_df, ax=axs[ax_i])
    ax_i += 1
    hz_ax = plot_sensor_band_training(lowhz_df, centerhz_df, highhz_df, ax=axs[ax_i])

    if fig is None:
        fig = axs[0].get_figure()

    if title is not None:
        fig.suptitle(title, fontsize=15)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, {'loss_ax': loss_ax, 'hz_ax': hz_ax}

def make_hz_frame_from_results(results):
    lowhz_df = pd.read_json(results['low_hz_frame']).sort_index()
    highhz_df = pd.read_json(results['high_hz_frame']).sort_index()
    centerhz_df = (highhz_df + lowhz_df) / 2.
    return lowhz_df, centerhz_df, highhz_df


def make_loss_frame_from_results(results):
    df = pd.DataFrame(results['batch_losses']).T
    df.index.name = 'epoch'
    return df


def run_one(options, result_file):
    ###############
    ### Path handling
    result_base_path, result_filename = os.path.split(result_file)
    result_id = result_filename.split('.')[0]
    # Load results to get the file name of the model
    results = json.load(open(result_file))
    if options.eval_filter is not None:
        should_continue = eval(options.eval_filter, dict(r=results))
        if not should_continue:
            print("Skipping result at %s because filter returned False" % result_file)
            return None


    model_filename = os.path.split(results['save_model_path'])[-1]
    base_model_path = options.base_model_path
    if base_model_path is None:
        base_model_path = os.path.join(result_base_path, 'models')
        print("Base model path not give - assuming path '%s'" % base_model_path)

    base_output_path = result_id if options.base_output_path is None else options.base_output_path
    from pathlib import Path
    print(f"Creating results dir {base_output_path} if it doesn't already exist")
    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    ###############
    ### Processing
    model_kws = results['model_kws']
    #loss_df = pd.DataFrame(results['batch_losses']).T
    loss_df = make_loss_frame_from_results(results)
    lowhz_df, centerhz_df, highhz_df = make_hz_frame_from_results(results)
    kwarg_str = ", ".join(["%s=%s" % (str(k), str(v)) if (i % 5) or i ==0 else "\n%s=%s" % (str(k), str(v))
                          for i, (k, v) in enumerate(results['model_kws'].items())])
    perf_str = '|| '.join(['%s=%s' % (str(k), str(np.round(results[k], 3))) for k in ['accuracy', 'precision', 'recall']])
    title = (f"({results['uid']})\ntrain:[{results['train_sets']}] || cv:[{results['cv_sets']}] || test:[{results['test_sets']}] \n\
    Num Params={results['num_params']} || {perf_str}\n\
    {results['model_name']}({kwarg_str})")
    #print(title)
    #title = f"Model {results['']}"
    fig, ax_map = multi_plot_training(loss_df, lowhz_df, centerhz_df, highhz_df, title=None)
    ax_map['loss_ax'].set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(base_output_path, "training_plots.pdf"))

    if options.eval_sets is not None:
        model_path = os.path.join(base_model_path, model_filename)
        print("Loading model located at: " + str(model_path))

        # Handle if the user puts in train/cv/test, otherwise use the string as given
        eval_set_str = {k: results[k + "_sets"] for k in ["test", "cv", "train"]}.get(options.eval_sets,
                                                                                      options.eval_sets)

        dataset_cls = datasets.BaseDataset.get_dataset_by_name(results['dataset'])
        data_k_l = dataset_cls.make_tuples_from_sets_str(eval_set_str)
        dset = dataset_cls(patient_tuples=data_k_l,
                           # TODO: This may hide problems or cause issues?
                           sensor_columns=list(range(model_kws['in_channels'])))

        if results['model_name'] == 'base-sn':
            model = base.BaseMultiSincNN(**model_kws)
        elif results['model_name'] == 'tnorm-base-sn':
            model = base.TimeNormBaseMultiSincNN(**model_kws)
        elif results['model_name'] == 'base-cnn':
            model = base.BaseCNN(**model_kws)
        else:
            raise ValueError(f"Unrecognized model_name: {results['model_name']} in {result_file})")


        with open(model_path, 'rb') as f:
            model_state = torch.load(f)

        model.load_state_dict(model_state)
        model.to(options.device)

        preds_map = dset.eval_model(model, options.eval_win_step_size,
                                    device=options.device)

        for ptuple, data_map in dset.data_maps.items():
            print("Plotting " + str(ptuple))
            ptuple_str = "-".join(str(v) for v in ptuple)
            fig, ax = plot_model_preds(preds_s=preds_map[ptuple], data_map=data_map,
                                       sample_index_map=dset.sample_index_maps[ptuple])
            fig_filename = os.path.join(base_output_path, "prediction_plot_for_%s.pdf" % ptuple_str)
            print("Saving to " + str(fig_filename))
            fig.savefig(fig_filename)


    # TODO: return something useful - dictionary of results? A class of results?j

def run(options):
    if options.result_file is None:
        raise ValueError("Must provide result-file option")

    from glob import glob
    result_files = list(glob(options.result_file))
    if len(result_files) == 1:
        return run_one(options, result_files[0])

    print(f"Producing results on {len(result_files)} result files")
    original_base_path = str(options.base_output_path)
    for r in tqdm(result_files):
        fname = os.path.split(r)[-1].split('.')[0]
        options.base_output_path = os.path.join(original_base_path, fname)
        res = run_one(options, r)

    #[run_one(options, r) for r in tqdm(result_files)]


default_option_kwargs = [
    dict(dest="--result-file", default=None, type=str),
    dict(dest="--base-model-path", default=None, type=str),
    dict(dest='--eval-sets', default=None, type=str,
         help="Dataset to run the loaded model against - use train/cv/test for the data used to build the model"
              "or specify the dataset name (e.g. MC-19-0. If unset, model will not be evaluated on data"),

    dict(dest="--eval-win-step-size", default=1, type=int),
    dict(dest="--base-output-path", default=None, type=str),
    dict(dest="--eval-filter", default=None, type=str),
    dict(dest='--device', default='cuda:0'),
]
if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs,
                                  description="ASPEN+MHRG Result Parsing")
    m_options = parser.parse_args()
    results = run(m_options)

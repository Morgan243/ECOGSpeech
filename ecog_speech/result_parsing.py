import pandas as pd
import numpy as np
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from glob import glob
import os
import json
from ecog_speech import datasets, feature_processing, experiments, utils
from ecog_speech.models import base
from tqdm.auto import tqdm
import torch


def frame_to_torch_batch(_df, win_size, win_step):
    _arr = torch.from_numpy(_df.values)
    outputs =list()
    for _iix in range(0, _arr.shape[0] - win_size, win_step):
        _ix = slice(_iix, _iix + win_size)
        outputs.append(_arr[_ix].unsqueeze(0))
    return torch.cat(outputs).permute(0, 2, 1)


def make_outputs(sn_model, in_batch_arr):
    sn_model.eval()
    with torch.no_grad():
        sn_out = sn_model.m[:3](in_batch_arr)
        out = sn_model.m(in_batch_arr)
    return sn_out, out


def swap_tdelta_to_total_seconds_index(df):
    return (df.rename_axis(index='tdelta').reset_index()
            .pipe(lambda _df: _df.assign(Seconds=_df['tdelta'].dt.total_seconds())).set_index('Seconds').drop('tdelta', axis=1))


def wrangle_and_plot_pred_inspect(model, nww: datasets.NorthwesternWords, wrd_ix: int,
                                  sens_to_plt=None, patient_tuple=None):
    import seaborn as sns

    #test_nww_word_id = {mname: wrd_ix for mname, _nww in test_nww_map.items()}
    # test_nww_sample_ix_maps, test_pos_wrd_ix_l_map, test_neg_wrd_ix_l_map = map_model_words(test_nww_map, test_nww_word_id)
    # samp_ix_map = t_nww.sample_index_maps[next(iter(_t_nww.sample_index_maps.keys()))]
    patient_tuple = next(iter(nww.data_maps.keys())) if patient_tuple is None else patient_tuple
    data_map = nww.data_maps[patient_tuple]
    samp_ix_map = next(iter(nww.sample_index_maps.values()))
    pos_win_ixes, neg_win_ixes = samp_ix_map[wrd_ix], samp_ix_map[-wrd_ix]

    ###----
    left_pad_t, right_pad_t = pd.Timedelta('1500ms'), pd.Timedelta('200ms')

    pos_start, pos_end = pos_win_ixes[0].min(), pos_win_ixes[-1].max()
    neg_start, neg_end = neg_win_ixes[0].min(), neg_win_ixes[-1].max()

    plt_slice = slice(pos_start - left_pad_t, neg_end + right_pad_t)
    ###-----
    ecog_win_df = data_map['ecog'].loc[plt_slice]
    audio_win_s = data_map['audio'].loc[plt_slice].rename('audio')
    stim_win_s = data_map['stim'].loc[plt_slice].rename('stim')

    contig_ix = ecog_win_df.index

    contig_ecog_arr = torch.from_numpy(ecog_win_df.values)
    with torch.no_grad():
        contig_sn_out = model.m[:3](contig_ecog_arr.transpose(0, 1).unsqueeze(0))[0].detach().numpy()
    ###----

    t_ix = contig_ix[model.window_size:]

    t_ecog_arr = frame_to_torch_batch(ecog_win_df, model.window_size, 1)

    sn_out, model_preds = make_outputs(model, t_ecog_arr)

    # sn_out, out = make_outputs(t_model, t_nww[0]['ecog_arr'].unsqueeze(0))

    model_pred_s = pd.Series(model_preds.squeeze().detach().numpy(), index=t_ix, name='model_pred_proba')

    model_pred_s.rename_axis(index='ts', inplace=True)
    ##-----
    # Hilbert (envelope) of each band-sensor time series
    # - Maps sensor id to a dataframe with index as time and columns of band wit envelope values
    sens_band_hilb_df_map = {
        s_i: pd.concat([feature_processing.make_hilbert_df(pd.Series(_arr, name=b_i)).envelope.rename(b_i)
                        for b_i, _arr in enumerate(band_arr)], axis=1)
        for s_i, band_arr in enumerate(contig_sn_out)}
    # Concat all hilbert data together into a Frame
    # - Now multi-index with levels of sensor-time
    contig_hilbert_df = pd.concat([band_hil_df.assign(sensor=s_i, ts=contig_ix)
                                   for s_i, band_hil_df in sens_band_hilb_df_map.items()]).set_index(['sensor', 'ts'])
    contig_hilbert_df.columns.rename('envelope', inplace=True)

    ##-----
    n_largest_target_corr_df = contig_hilbert_df.apply(
        lambda s: s.unstack().T.join(model_pred_s).corr()[model_pred_s.name].drop([model_pred_s.name]).abs().nlargest())

    if sens_to_plt is None:
        sens_to_plt = n_largest_target_corr_df.notnull().sum(1).nlargest().index.tolist()

    # n_largest_target_corr_df
    ##------
    fs = 600
    fft_freq = np.fft.rfftfreq(300, 1.0 / fs)
    ##-----
    plt_sens_fft_arr = torch.fft.rfft(sn_out).abs()[:, sens_to_plt, :, :]
    # plt_sens_fft_arr = torch.fft.rfft(t_ecog_arr).abs()[:, sens_to_plt, :]
    plt_avg_fft_arr = plt_sens_fft_arr.mean(1)  # .squeeze()

    # plt_avg_fft_df = pd.DataFrame(plt_avg_fft_arr.detach().numpy(), index=t_ix, columns=fft_freq)
    ##-----

    plt_audio_s = swap_tdelta_to_total_seconds_index(audio_win_s)

    plt_out_s = swap_tdelta_to_total_seconds_index(model_pred_s)[model_pred_s.name]

    plt_stim_s = swap_tdelta_to_total_seconds_index(stim_win_s)[stim_win_s.name]

    ##--------------
    sub_bands = len(contig_hilbert_df.columns)
    nrows = 1 + sub_bands * 2
    plt_ratio = 1. / sub_bands
    print("Nrows " + str(nrows))
    fig, axs = matplotlib.pyplot.subplots(nrows=nrows, figsize=(18, 3.1 * (4)),
                                          # sharex='col',
                                          constrained_layout=True,
                                          gridspec_kw={'height_ratios': ([plt_ratio] * sub_bands * 2) + [1.3]},
                                          squeeze=False)
    # cbar_ax = fig.add_axes([1, .76, .01, .2])
    axs = axs.reshape(-1)
    max_y = 1.2
    ###---
    ax_i = 0
    for i, c in enumerate(contig_hilbert_df.columns.tolist()):
        ax = sns.heatmap(contig_hilbert_df.loc[sens_to_plt][[c]].unstack(0).rolling(15).mean().T, annot=False,
                         robust=True, cbar=True,
                         cmap='inferno',
                         # cmap='cividis',
                         # cmap='jet',
                         ax=axs[ax_i],  # cbar_ax=cbar_ax
                         )
        ax.set_xticks([])
        ax.set_xlabel('')
        # if i==0:
        if contig_hilbert_df.shape[1] == 1 or contig_hilbert_df.shape[1] // 2 == i:
            ax.set_ylabel('Band-Sensor', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(labelsize=11)

        #
        ax_i += 1

    low_hz = np.abs(model.get_band_params()[0]['low_hz'] * 600)[:, 0]  # .squeeze()
    band_hz = np.abs(model.get_band_params()[0]['band_hz'] * 600)[:, 0]  # .squeeze()
    print("Low hz shape: " + str(model.get_band_params()[0]['low_hz'].shape))
    for i in range(plt_avg_fft_arr.shape[1]):
        plt_avg_fft_df = pd.DataFrame(plt_avg_fft_arr.select(1, i).detach().numpy(), index=t_ix, columns=fft_freq)
        hz_slice = slice(np.floor(low_hz[i]), np.ceil(low_hz[i] + max(band_hz[i], 3)))
        print(hz_slice)
        ax = sns.heatmap(plt_avg_fft_df.rolling(15).mean().T.loc[hz_slice].sort_index(ascending=False), annot=False,
                         robust=True, cbar=True,
                         cmap='viridis',
                         # cmap='cividis',
                         # cmap='jet',
                         ax=axs[ax_i],  # cbar_ax=fig.add_axes([1, .52, .01, .2])
                         )
        ax.set_xticks([])
        ax.set_xlabel('')
        if plt_avg_fft_arr.shape[1] == 1 or plt_avg_fft_arr.shape[1] // 2 == i:
            ax.set_ylabel('Spectra (Hz)', fontsize=10)
        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(labelsize=11)

        ax_i += 1

    ## Model prediction with error fill
    model_mean = plt_out_s.rolling(15).mean()
    model_std = plt_out_s.rolling(15).std()

    ax = model_mean.plot(ax=axs[ax_i], color='tab:blue', alpha=1, lw=3.5, label='Model E(P(Speaking))')
    ax.fill_between(plt_out_s.index, model_mean + model_std, model_mean - model_std, alpha=0.2, color='tab:blue')

    ## Mark where stimulus cue was
    ax.vlines(plt_stim_s[plt_stim_s == wrd_ix].index.min(), 0, max_y, color='tab:purple', lw=5, label='Stimulus Onset',
              ls='-.')

    ## Aggregat audio a bit - so many samples makes the plot complex/time consuming
    ax2 = ax.twinx()
    ax2 = plt_audio_s.rolling(10).mean().plot(ax=ax2, color='grey', alpha=0.5, label='Speech Waveform', legend=False)
    ax2.set_yticks([])  # don't care about audio value?

    ## Speaking Region
    ax.fill_betweenx([0, max_y], pos_start.total_seconds(), pos_end.total_seconds(),
                     color='tab:green', alpha=0.3, label='Labeled Speaking Region')

    # ax.vlines(pos_start.total_seconds(), 0, 1, color='tab:green', lw=5, ls='-', label='Start of First Speaking Window')
    ax.vlines(pos_win_ixes[0].max().total_seconds(), 0, max_y, color='tab:green', lw=5, ls='--',
              label='End of First Speaking Window')

    ax.fill_betweenx([0, max_y], neg_start.total_seconds(), neg_end.total_seconds(),
                     color='tab:orange', alpha=0.3, label='Labeled Non-Speaking Region')

    ax.vlines(neg_win_ixes[0].max().total_seconds(), 0, max_y, color='tab:orange', lw=5, ls='--',
              label='End of First Non-Speaking Window')

    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, .25, .5, .75, 1])

    ax.set_xlim(plt_out_s.index.min(), plt_out_s.index.max())
    ax.set_ylabel('P(Speaking)', fontsize=15)
    ax.set_xlabel("Trial Time (seconds from start)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, which='both', alpha=0.25, lw=3)

    fig.legend(ncol=4, loc=(0.125, 0.35), framealpha=1, fontsize=13, handlelength=4)
    fig.suptitle("Word code " + str(wrd_ix) + ' - "' + data_map['word_code_d'][wrd_ix] + '"')
    return fig, axs

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


def plot_training(loss_df, title=None, ax=None, logy=True, **plt_kwargs):
    ax = loss_df.plot(figsize=(6, 5), grid=True, lw=3, ax=ax,
                      style=['-'] + (['--'] * (loss_df.shape[-1] - 1)), logy=logy, **plt_kwargs)
    ax.set_ylabel('Loss Value', fontsize=13)
    ax.set_xlabel('Epoch', fontsize=13)
    if title is not None:
        ax.set_title(title, fontsize=15)
    return ax


def plot_sensor_band_training(lowhz_df, centerhz_df, highhz_df,
                              title=None, ax=None, figsize=(15, 6)):
    for c in lowhz_df.columns:
        ax = centerhz_df[c].plot(figsize=figsize, lw=3, ax=ax)
        ax.fill_between(centerhz_df.index,
                        lowhz_df[c],
                        highhz_df[c],
                        alpha=0.5)
    if title is not None:
        ax.set_title(title, fontsize=15)
    ax.set_ylabel('Hz', fontsize=13)
    # TODO: Can we map this to actual batches?
    ax.set_xlabel('Batch Sample Index', fontsize=13)
    ax.axhline(0, lw=3, color='grey')
    ax.grid(True)
    #plt.clf()
    return ax


def multi_plot_training(loss_df, lowhz_df, centerhz_df,  highhz_df,
                        title=None, figsize=(15, 6),
                        axs=None):
    fig = None
    if axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=figsize)
    axs = axs.reshape(-1)

    ax_i = 0
    loss_ax = plot_training(loss_df, ax=axs[ax_i])
    ax_i += 1
    if lowhz_df is not None:
        hz_ax = plot_sensor_band_training(lowhz_df, centerhz_df, highhz_df,
                                          ax=axs[ax_i], figsize=figsize)
    else:
        hz_ax = axs[ax_i]
        hz_ax.annotate("No SincNet parameters available in results", (0.3, 0.3))

    if fig is None:
        fig = axs[0].get_figure()

    if title is not None:
        fig.suptitle(title, fontsize=15)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, {'loss_ax': loss_ax, 'hz_ax': hz_ax}


def plot_model_overview(results):
    loss_df = make_loss_frame_from_results(results)
    lowhz_df, centerhz_df, highhz_df = make_hz_frame_from_results(results)
    kwarg_str = ", ".join(["%s=%s" % (str(k), str(v)) if (i % 5) or i ==0 else "\n%s=%s" % (str(k), str(v))
                          for i, (k, v) in enumerate(results['model_kws'].items())])
    perf_str = '|| '.join(['%s=%s' % (str(k), str(np.round(results[k], 3))) for k in ['accuracy', 'f1', 'precision', 'recall']])
    title = (f"({results['uid']})\ntrain:[{results['train_sets']}] || cv:[{results['cv_sets']}] || test:[{results['test_sets']}] \n\
    Num Params={results['num_params']} || {perf_str}\n\
    {results['model_name']}({kwarg_str})")
    #print(title)
    #title = f"Model {results['']}"
    fig, ax_map = multi_plot_training(loss_df, lowhz_df, centerhz_df, highhz_df, title=None)
    ax_map['loss_ax'].set_title(title, fontsize=13)
    ax_map['loss_ax'].axvline(results['best_model_epoch'], ls='--', color='black')
    fig.tight_layout()
    #fig.savefig(os.path.join(base_output_path, "training_plots.pdf"))
    return fig, ax_map


####
def load_results_to_frame(p):
    #base_path = "../ecog_speech/results_per_patient_sn_2105_50epochs/"
    #result_files = glob(os.path.join(p, '*.json'))
    result_files = glob(p)

    json_result_data = [json.load(open(f)) for f in tqdm(result_files)]
    return pd.DataFrame(json_result_data)


def plot_agg_performance(results_df):
    import seaborn as sns

    # Choose a metric
    perf_col = 'f1'

    #performance_cols = ['accuracy', 'f1', 'precision', 'recall']
    config_params = ['model_name', 'dataset', 'dense_width',
                     'sn_n_bands', 'sn_kernel_size', 'sn_padding',
                     'bw_reg_weight', 'cog_attn', 'shuffle_channels',
                     'n_cnn_filters', 'dropout', 'dropout_2d', 'in_channel_dropout_rate',
                     'batchnorm', 'roll_channels', 'power_q', 'n_epochs']

    results_df['bw_reg_weight'] = results_df['bw_reg_weight'].fillna(-1)
    results_df['test_patient'] = results_df['test_sets'].str.split('-').apply(lambda l: '-'.join(l[:-1]))
    results_df['test_fold'] = results_df['test_sets'].str.split('-').apply(lambda l: l[-1])

    nun_config_params = results_df[config_params].nunique()

    config_cols = nun_config_params[nun_config_params > 1].index.tolist()
    fixed_config_cols = nun_config_params[nun_config_params == 1].index.tolist()
    print(f"Fixed Params: {', '.join(fixed_config_cols)}")
    print(f"Changing Params: {', '.join(config_cols)}")

    grp = results_df.groupby(config_cols + ['test_patient'], dropna=False)[perf_col]
    res_perf = grp.mean()
    res_n = grp.size().rename('N')

    res_perf_df = res_perf.reset_index()
    res_n_df = res_n.reset_index()

    def hplot(*args, **kwargs):
        # print(args)
        x = kwargs.pop('data')
        plt_df = x.groupby(list(args[:-1])).mean().reset_index().pivot(*args)
        # display(plt_df)
        ax = sns.heatmap(plt_df.T,
                         annot=True, **kwargs)
        return ax

    extra_kws = dict()
    if 'in_channel_dropout_rate' in res_perf_df.columns:
        extra_kws = dict(row='in_channel_dropout_rate')

    figs, axes = list(), list()
    try:
        g = sns.FacetGrid(res_perf_df, col="test_patient",
                          #row='in_channel_dropout_rate',
                          # Sharing axis doesn't seem to work so well with sparse results - axes names and ticks get weird
                          sharex=False, sharey=False, height=5,
                          **extra_kws)
        sns_fg = g.map_dataframe(hplot, 'n_cnn_filters', 'sn_n_bands', 'f1',
                        cmap='Greens', vmax=1., vmin=0.55,
                        cbar=False, linewidths=1, linecolor='grey')
        figs.append(sns_fg.fig)
        axes.append(sns_fg.axes)
    except KeyError as e:
        print("Cant plot full heatmap: " + str(e))

    fig, axs = matplotlib.pyplot.subplots(ncols=2, figsize=(9, 6))
    ax = res_n.plot.barh(ax=axs[0], grid=True, title='model config N', color='grey')
    ax.set_xlabel('N experiments (x folds)')
    ax = res_perf.plot.barh(ax=axs[1], grid=True, title='model config performance')
    ax.set_xlabel(f'{perf_col} score')
    fig.tight_layout()
    figs.append(fig)
    axes.append(ax)

    return figs, axes


def make_hz_frame_from_results(results):
    lowhz_df = highhz_df = centerhz_df = None
    if 'low_hz_frame' in results:
        lowhz_df = pd.read_json(results['low_hz_frame']).sort_index().abs()
        highhz_df = pd.read_json(results['high_hz_frame']).sort_index().abs()
        centerhz_df = (highhz_df + lowhz_df) / 2.
    return lowhz_df, centerhz_df, highhz_df


def make_loss_frame_from_results(results):
    df = pd.DataFrame(results['batch_losses']).T
    df.index.name = 'epoch'
    return df


def run_one(options, result_file):
    output_fig_map = dict()
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

    model_kws = results['model_kws']
    base_model_path = options.base_model_path
    if base_model_path is None:
        base_model_path = os.path.join(result_base_path, 'models')
        print("Base model path not give - assuming path '%s'" % base_model_path)

    base_output_path = result_id if options.base_output_path is None else options.base_output_path
    print(f"Creating results dir {base_output_path} if it doesn't already exist")
    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    ###############
    ### Processing
    fig, ax_map = plot_model_overview(results)
    fig.savefig(os.path.join(base_output_path, "training_plots.pdf"))
    output_fig_map['training_plots'] = fig

    if options.eval_sets is not None:
        model_filename = os.path.split(results['save_model_path'])[-1]
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

        dl_map = dset.to_eval_replay_dataloader(win_step=options.eval_win_step_size)
        preds_map = base.Trainer.generate_outputs_from_model(model, dl_map, device=options.device,)
        #preds_map = dset.eval_model(model, options.eval_win_step_size,
        #                            device=options.device)

        for ptuple, data_map in dset.data_maps.items():
            print("Plotting " + str(ptuple))
            ptuple_str = "-".join(str(v) for v in ptuple)
            if options.pred_inspect_eval:
                all_wrds_codes = [k for k in next(iter(dset.sample_index_maps.values())).keys() if k > 0]
                from matplotlib.backends.backend_pdf import PdfPages
                pp = PdfPages('stim_code_preds_MC24_4band_0602.pdf')
                fig, ax_map = plot_model_overview(results)
                fig.savefig(pp, format='pdf')
                for t_wrd in tqdm(all_wrds_codes):
                    #fig, axs = wran(t_model, t_dmap, wrd_ix=t_wrd)
                    fig, axs = wrangle_and_plot_pred_inspect(model, dset, t_wrd, patient_tuple=ptuple)
                    fig.savefig(pp, format='pdf')
                    ## TODO: Close figure here?
                    matplotlib.pyplot.close(fig)
                pp.close()
                fig_name = "prediction_inspect_plot_for_%s.pdf" % ptuple_str
            else:
                fig, ax = plot_model_preds(preds_s=preds_map[ptuple], data_map=data_map,
                                           sample_index_map=dset.sample_index_maps[ptuple])
                fig_name = "prediction_plot_for_%s.pdf" % ptuple_str
            fig_filename = os.path.join(base_output_path, fig_name)
            print("Saving to " + str(fig_filename))
            fig.savefig(fig_filename)
            output_fig_map[fig_name] = fig

    # TODO: return something useful - dictionary of results? A class of results?j
    return results, output_fig_map


def run(options):
    if options.result_file is None:
        raise ValueError("Must provide result-file option")

    from glob import glob
    result_files = list(glob(options.result_file))
    if len(result_files) == 1:
        return run_one(options, result_files[0])

    print(f"Producing results on {len(result_files)} result files")
    base_p = os.path.join(os.path.split(options.result_file)[0], 'parsed/')
    original_base_path = base_p if options.base_output_path is None else options.base_output_path

    res_map = dict()
    for r in tqdm(result_files):
        fname = os.path.split(r)[-1].split('.')[0]
        options.base_output_path = os.path.join(original_base_path, fname)
        ro = run_one(options, r)
        if ro is not None:
            res_map[r] = ro

    #if options.training_and_perf_path:
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(os.path.join(base_p, 'training_and_perf.pdf'))
    results_df = pd.DataFrame([r for r, output_figs in res_map.values()])
    figs, axes = plot_agg_performance(results_df)
    for fig in figs:
        fig.savefig(pp, format='pdf')

    sorted_res_map = sorted(list(res_map.items()), key=lambda _results: _results[1][0]['f1'], reverse=True)
    for r, (results, output_figs) in sorted_res_map:
        output_figs['training_plots'].savefig(pp, format='pdf')
    pp.close()

    #[run_one(options, r) for r in tqdm(result_files)]


default_option_kwargs = [
    dict(dest="--result-file", default=None, type=str, required=True),
    dict(dest="--base-model-path", default=None, type=str),
    dict(dest='--eval-sets', default=None, type=str,
         help="Dataset to run the loaded model against - use train/cv/test for the data used to build the model"
              "or specify the dataset name (e.g. MC-19-0. If unset, model will not be evaluated on data"),
    #dict(dest="--training-and-perf-path", default=None, type=str),
    dict(dest="--eval-win-step-size", default=1, type=int),
    dict(dest="--pred-inspect-eval", default=False, action='store_true'),
    dict(dest="--base-output-path", default=None, type=str),
    dict(dest="--eval-filter", default=None, type=str),
    dict(dest='--device', default='cuda:0'),
]
if __name__ == """__main__""":
    parser = utils.build_argparse(default_option_kwargs,
                                  description="ASPEN+MHRG Result Parsing")
    m_options = parser.parse_args()
    m_results = run(m_options)

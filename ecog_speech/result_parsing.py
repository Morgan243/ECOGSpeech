import pandas as pd
import numpy as np
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from glob import glob
import os
import json

from ecog_speech import datasets, feature_processing, experiments, utils
from ecog_speech import models
from ecog_speech.models import base
from tqdm.auto import tqdm
import torch

from pprint import pprint

def frame_to_torch_batch(_df, win_size, win_step):
    _arr = torch.from_numpy(_df.values)
    outputs =list()
    for _iix in range(0, _arr.shape[0] - win_size, win_step):
        _ix = slice(_iix, _iix + win_size)
        outputs.append(_arr[_ix].unsqueeze(0))
    return torch.cat(outputs).permute(0, 2, 1)


def make_outputs(sn_model, in_batch_arr, device=None):
    sn_model.eval()
    if device is not None:
        sn_model = sn_model.to(device)
        in_batch_arr = in_batch_arr.to(device)

    with torch.no_grad():
        sn_out = sn_model.m[:3](in_batch_arr)
        out = sn_model.m(in_batch_arr)
    return sn_out, out


def swap_tdelta_to_total_seconds_index(df):
    return (df.rename_axis(index='tdelta').reset_index()
            .pipe(lambda _df: _df.assign(Seconds=_df['tdelta'].dt.total_seconds())).set_index('Seconds').drop('tdelta', axis=1))


def wrangle_and_plot_pred_inspect(model, nww: datasets.NorthwesternWords, wrd_ix: int,
                                  sens_to_plt=None, patient_tuple=None, device=None):
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
    if device is not None:
        model = model.to(device)
        contig_ecog_arr = contig_ecog_arr.to(device)

    with torch.no_grad():
        contig_sn_out = model.m[:3](contig_ecog_arr.transpose(0, 1).unsqueeze(0))[0].cpu().detach().numpy()
    ###----

    t_ix = contig_ix[model.window_size:]

    t_ecog_arr = frame_to_torch_batch(ecog_win_df, model.window_size, 1)

    sn_out, model_preds = make_outputs(model, t_ecog_arr, device=device)

    # sn_out, out = make_outputs(t_model, t_nww[0]['ecog_arr'].unsqueeze(0))

    model_pred_s = pd.Series(model_preds.squeeze().cpu().detach().numpy(), index=t_ix, name='model_pred_proba')

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
        plt_avg_fft_df = pd.DataFrame(plt_avg_fft_arr.select(1, i).cpu().detach().numpy(), index=t_ix, columns=fft_freq)
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
                              title=None, ax=None, peak_fs=300, figsize=(15, 6),
                              **plt_kws):
    for c in lowhz_df.columns:
        ax = centerhz_df[c].plot(figsize=figsize, lw=3, ax=ax, **plt_kws)
        ax.fill_between(centerhz_df.index,
                        lowhz_df[c],
                        highhz_df[c],
                        alpha=0.5)
    if title is not None:
        ax.set_title(title, fontsize=15)
    ax.set_ylabel('Hz', fontsize=13)
    # TODO: Can we map this to actual batches?
    ax.set_xlabel('Batch Sample Index', fontsize=13)
    ax.axhline(peak_fs, lw=3, color='grey', alpha=0.4)
    ax.axhline(0, lw=3, color='grey', alpha=0.4)
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

    perf_measures = ['accuracy', 'f1', 'precision', 'recall']

    train_perf_str = '|| '.join(['%s=%s' % (str(k), str(np.round(results['train_'+k], 3))) for k in perf_measures])
    cv_perf_str = '|| '.join(['%s=%s' % (str(k), str(np.round(results['cv_'+k], 3))) for k in perf_measures])
    perf_str = '|| '.join(['%s=%s' % (str(k), str(np.round(results[k], 3))) for k in perf_measures])

    if results.get('random_labels', False):
        perf_str += ' !! RANDOM LABELS  !!'
    title = (f"({results['name']})\ntrain:[{results['train_sets']}] || cv:[{results['cv_sets']}] || test:[{results['test_sets']}] \n\
    Num Params={results['num_params']} || {results['model_name']}({kwarg_str})\nTEST:{perf_str}\nCV:{cv_perf_str}\nTrain:{train_perf_str}\n\
    ")
    #print(title)
    #title = f"Model {results['']}"
    fig, ax_map = multi_plot_training(loss_df, lowhz_df, centerhz_df, highhz_df, title=None)
    ax_map['loss_ax'].set_title(title, fontsize=13)
    ax_map['loss_ax'].axvline(results['best_model_epoch'], ls='--', color='black')
    fig.tight_layout()
    #fig.savefig(os.path.join(base_output_path, "training_plots.pdf"))
    return fig, ax_map


####
def load_results_to_frame(p, config_params=None):
    result_files = glob(p)

    json_result_data = [json.load(open(f)) for f in tqdm(result_files)]
    results_df = pd.DataFrame(json_result_data)
    #results_df['bw_reg_weight'] = results_df['bw_reg_weight'].fillna(-1)
    try:
        results_df['test_patient'] = results_df['test_sets'].str.split('-').apply(lambda l: '-'.join(l[:-1]))
        results_df['test_fold'] = results_df['test_sets'].str.split('-').apply(lambda l: l[-1])
    except:
        print("Unable to parse test patient - was there one?")


    ####
    if config_params is None:
        return results_df
    elif isinstance(config_params, bool) and config_params:
        config_params = [n for n in experiments.all_model_hyperparam_names if n in results_df.columns.values]

    print("All config params to consider: " + ", ".join(config_params))
    #config_params = default_config_params if config_params is None else config_params
    nun_config_params = results_df[config_params].nunique()

    config_cols = nun_config_params[nun_config_params > 1].index.tolist()
    fixed_config_cols = nun_config_params[nun_config_params == 1].index.tolist()

    ###

    try:
        fixed_unique = results_df[fixed_config_cols].apply(pd.unique)
        if isinstance(fixed_unique, pd.DataFrame):
            fixed_d = fixed_unique.iloc[0].to_dict()
        else:
            fixed_d = fixed_unique.to_dict()

        fixed_d_str = "\n\t".join(f"{k}={v}" for k, v in fixed_d.items())
        #print(f"Fixed Params: {', '.join(fixed_config_cols)}")
        print(f"Fixed Params:\n------------\n\t{fixed_d_str}")
        print(f"Changing Params: {', '.join(config_cols)}\n-------------\n")
        print(results_df.groupby(config_cols).size().unstack(-1))
    except:
        print("Unable to summarize parameterization of result files... new result structure?")

    return fixed_config_cols, config_cols, results_df


def plot_param_perf_facet(results_df):
    import seaborn as sns
    sns.set(font_scale=2.5, style='whitegrid')
    #results_df = sn_results_df
    rand_res_map = results_df[results_df.random_labels].groupby('test_patient').accuracy.mean().to_dict()
    g = sns.catplot(data=results_df[(~results_df.random_labels)
                                    & results_df.batchnorm
                                    & (~results_df.shuffle_channels)
                                    & results_df.sn_n_bands.ne(2)
                                    # & results_df.in_channel_dropout_rate.eq(0)
                                    #& results_df.dropout.eq(.25)
                                    ],
                    x='sn_n_bands', y='accuracy', hue='in_channel_dropout_rate', palette='mako',
                    col_order=['MC-19', 'MC-21', 'MC-22', 'MC-24',  # 'MC-25',
                               'MC-26'],
                    col='test_patient',   row='dropout',
                    kind='bar', height=6.5, aspect=.7)
    ln = None
    for tp, v in rand_res_map.items():
        if tp in g.axes_dict:
            ln = g.axes_dict[tp].axhline(v, lw=9, ls='--', color='xkcd:tomato', alpha=0.75)
    # g.fig.legend([ln], ['Random'], 'lower right', frameon=False)

    if ln is not None:
        g.fig.legend([ln], ['Performance on\nRandom Target'], 'lower right', frameon=False, ncol=2, fontsize=30)
    g._legend.set_title("Input Channel Dropout", )
    #g._legend

    # g.set_titles(template='{col_name}')
    g.set_titles(template='Patient {col_name}', fontsize=40)

    g.set_xlabels('Num Bands')
    g.set_ylabels('Accuracy')
    g.set(ylim=(0.47, 1.), yticks=np.arange(0.5, 1.01, 0.1))
    return g.fig


def plot_agg_performance(results_df):
    import seaborn as sns

    # Choose a metric
    perf_col = ['f1', 'accuracy']

    #performance_cols = ['accuracy', 'f1', 'precision', 'recall']
    #config_params = ['model_name', 'dataset', 'dense_width',
    #                 'sn_n_bands', 'sn_kernel_size', 'sn_padding',
    #                 'bw_reg_weight', 'cog_attn', 'shuffle_channels',
    #                 'n_cnn_filters', 'dropout', 'dropout_2d', 'in_channel_dropout_rate',
    #                 'batchnorm', 'roll_channels', 'power_q', 'n_epochs', 'sn_band_spacing']

    print(results_df.columns)
    results_df['bw_reg_weight'] = results_df['bw_reg_weight'].fillna(-1)
    results_df['test_patient'] = results_df['test_sets'].str.split('-').apply(lambda l: '-'.join(l[:-1]))
    results_df['test_fold'] = results_df['test_sets'].str.split('-').apply(lambda l: l[-1])

    config_params_in_results = [p for p in experiments.all_model_hyperparam_names
                                if p in results_df.columns.values]
    missing_config_params = list(set(experiments.all_model_hyperparam_names) - set(config_params_in_results))
    if len(missing_config_params) > 0:
        print("Missing config params: " + (", ".join(missing_config_params)))
        print("Are these older results?")
    nun_config_params = results_df[config_params_in_results].nunique()

    config_cols = nun_config_params[nun_config_params > 1].index.tolist()
    fixed_config_cols = nun_config_params[nun_config_params == 1].index.tolist()
    print(f"Fixed Params: {', '.join(fixed_config_cols)}")
    print(f"Changing Params: {', '.join(config_cols)}")

    #if 'train_sets' in config_cols:
    #    config_cols.remove('train_sets')
    #    config_cols = ['train_sets'] + config_cols
    #print("CONFIG COLS: " + str(config_cols))
    grp = results_df.groupby(['test_patient'] + config_cols, dropna=False)[perf_col]
    res_perf = grp.mean()
    res_std = grp.std()
    res_n = grp.size().rename('N')

    res_perf_df = res_perf.reset_index()
    res_std_df = res_std.reset_index()
    res_n_df = res_n.reset_index()

    def hplot(*args, **kwargs):
        # print(args)
        x = kwargs.pop('data')
        plt_df = x.groupby(list(args[:-1])).mean().reset_index().pivot(*args)
        # display(plt_df)
        #sns.set(font_scale=1.7)
        ax = sns.heatmap(plt_df.T, annot_kws=dict(fontsize=18),
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

    fig, axs = matplotlib.pyplot.subplots(ncols=2,
                                          figsize=(9, max(len(res_n)*.25, 5)))
    ax = res_n.plot.barh(ax=axs[0], grid=True, title='model config N', color='grey')
    ax.set_xlabel('N experiments (x folds)')
    print("X ERROR")
    print(res_std_df)
    ax = res_perf.plot.barh(ax=axs[1], grid=True, title='model config performance', xerr=res_std)

    if isinstance(res_perf, pd.Series):
        ax.set_xlim((res_perf - res_std).min()*.95)
    elif isinstance(res_perf, pd.DataFrame):
        ax.set_xlim((res_perf - res_std).min().min()*.95)

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


def load_model_from_results(results, base_model_path=None, **kws_update):
    model_kws = results['model_kws']

    if base_model_path is not None:
        _p = results['save_model_path']
        _p = _p if '\\' not in _p else _p.replace('\\', '/')

        model_filename = os.path.split(_p)[-1]
        model_path = os.path.join(base_model_path, model_filename)
    else:
        model_path = results['save_model_path']


    #path = "C:\\temp\myFolder\example\\"
    #newPath = path.replace(os.sep, '/')

    #if base_model_path is None:
    #    if result_base_path is None:
    #        result_base_path = results['result_dir']
        #base_model_path = os.path.join(result_base_path, 'models')

#    if results['model_name'] == 'base-sn':
#        model = base.BaseMultiSincNN(**model_kws)
#    elif results['model_name'] == 'tnorm-base-sn':
#        model = base.TimeNormBaseMultiSincNN(**model_kws)
#    elif results['model_name'] == 'base-cnn':
#        model = base.BaseCNN(**model_kws)
#    else:
#        raise ValueError()
        #raise ValueError(f"Unrecognized model_name: {results['model_name']} in {result_file})")

    model_kws.update(kws_update)
    model, _ = models.make_model(model_name=results['model_name'], model_kws=model_kws)

    with open(model_path, 'rb') as f:
        model_state = torch.load(f)

    model.load_state_dict(model_state)
    return model
    #model.to(options.device)


def load_dataset_from_results(results, partition=None):
    all_parts = ["test", "cv", "train"]
    if partition is None:
        return {k: load_dataset_from_results(results, k)
                for k in all_parts if results[k+'_sets'] is not None}

    model_kws = results['model_kws']
    # Handle if the user puts in train/cv/test, otherwise use the string as given
    eval_set_str = {k: results[k + "_sets"] for k in all_parts}.get(partition,
                                                                                  partition)

    dataset_cls = datasets.BaseDataset.get_dataset_by_name(results['dataset'])
    data_k_l = dataset_cls.make_tuples_from_sets_str(eval_set_str)
    dset = dataset_cls(patient_tuples=data_k_l,
                       # TODO: This may hide problems or cause issues?
                       sensor_columns=list(range(model_kws['in_channels'])))
    return dset


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

        model, _ = models.make_model(model_name=results['model_name'], model_kws=model_kws)

        with open(model_path, 'rb') as f:
            model_state = torch.load(f)

        model.load_state_dict(model_state)
        model.to(options.device)

        #preds_map = dset.eval_model(model, options.eval_win_step_size,
        #                            device=options.device)

        #for ptuple, data_map in dset.data_maps.items():
        ptuple, data_map = next(iter(dset.data_maps.items()))
        print("Plotting " + str(ptuple))
        ptuple_str = "-".join(str(v) for v in ptuple)
        fig_name = "prediction_inspect_plot_for_%s.pdf" % ptuple_str
        fig_filename = os.path.join(base_output_path, fig_name)

        if options.pred_inspect_eval:
            all_wrds_codes = [k for k in next(iter(dset.sample_index_maps.values())).keys() if k > 0]
            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(fig_filename)
            fig, ax_map = plot_model_overview(results)
            fig.savefig(pp, format='pdf')
            for t_wrd in tqdm(all_wrds_codes):
                fig, axs = wrangle_and_plot_pred_inspect(model, dset, t_wrd, patient_tuple=ptuple,
                                                         device=options.device)
                fig.savefig(pp, format='pdf')
                ## TODO: Close figure here?
                #matplotlib.pyplot.close(fig)
                output_fig_map[ptuple_str + str(t_wrd)] = fig
            pp.close()
        else:
            dl_map = dset.to_eval_replay_dataloader(win_step=options.eval_win_step_size)
            win_size = results['model_kws']['window_size']
            win_step = options.eval_win_step_size
            t_preds_ix = data_map['ecog'].iloc[range(win_size, data_map['ecog'].shape[0], win_step)].index
            preds_map = base.Trainer.generate_outputs_from_model(model, dl_map, device=options.device,
                                                                 to_frames=True)
            fig, ax = plot_model_preds(preds_s=preds_map[ptuple].set_index(t_preds_ix)['preds'],
                                       data_map=data_map, sample_index_map=dset.sample_index_maps[ptuple])
            print("Saving to " + str(fig_filename))
            fig.savefig(fig_filename)
            output_fig_map[fig_name] = fig

    # TODO: return something useful - dictionary of results? A class of results?j
    return results, output_fig_map


def run(options):
    if options.result_file is None:
        raise ValueError("Must provide result-file option")

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

    fig = plot_param_perf_facet(results_df)
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

from dataclasses import dataclass
from typing import Optional

@dataclass
class ResultParsingOptions:
    result_file: str = None
    print_results: bool = False
    base_model_path: Optional[str] = None
    eval_sets: Optional[str] = None

    eval_win_step_size: int = 1
    pred_inspect_eval: bool = False
    base_output_path: Optional[str] = None
    eval_filter: Optional[str] = None
    device: str = 'cuda:0'

if __name__ == """__main__""":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(ResultParsingOptions, dest='result_parsing')
    args = parser.parse_args()
    result_parsing_options: ResultParsingOptions = args.result_parsing
    if result_parsing_options.print_results:
        result_files = list(glob(result_parsing_options.result_file))
        for rf in result_files:
            with open(rf) as f:
                results_json = json.load(f)
                print("--- " + rf + " ---")
                pprint(results_json)
    else:
        run(result_parsing_options)

#    parser = utils.build_argparse(default_option_kwargs,
#                                  description="ASPEN+MHRG Result Parsing")
#    m_options = parser.parse_args()
#    m_results = run(m_options)

import numpy as np
import pandas as pd
import matplotlib


def plot_word_sample_region(data_map, word_code=None, figsize=(15, 5), plot_features=False,
                            subplot_kwargs=None, feature_key='ecog', feature_ax=None, ax=None):
    word_code = np.random.choice(list(data_map['word_code_d'].keys())) if word_code is None else word_code

    t_silence_ixes = data_map['sample_index_map'][-word_code]
    t_speaking_ixes = data_map['sample_index_map'][word_code]

    silence_min_ix, silence_max_ix = t_silence_ixes[0].min(), t_silence_ixes[-1].max()
    speaking_min_ix, speaking_max_ix = t_speaking_ixes[0].min(), t_speaking_ixes[-1].max()

    padding = pd.Timedelta(.75, 's')

    plt_min = min(silence_min_ix, speaking_min_ix) - padding
    plt_max = max(silence_max_ix, speaking_max_ix) + padding
    plt_len = plt_max - plt_min

    #####
    plt_audio = (data_map['audio'].loc[plt_min:plt_max]
                 # .resample('5ms').first().fillna(method='ffill'),
                 .resample('5ms').median().fillna(0)
                 # .resample('5ms').interpolate().fillna(0)
                 )

    silence_s = pd.Series(0, index=plt_audio.index)
    silence_s.loc[silence_min_ix: silence_max_ix] = 0.95

    speaking_s = pd.Series(0, index=plt_audio.index)
    speaking_s.loc[speaking_min_ix: speaking_max_ix] = 0.95

    #####
    # feature_ax = None
    splt_kws = dict() if subplot_kwargs is None else subplot_kwargs
    if not plot_features and ax is None:
        fig, ax = matplotlib.pyplot.subplots(figsize=figsize, **splt_kws)
    elif not plot_features:
        fig = ax.get_figure()
    elif plot_features and ax is None or feature_ax is None:
        fig, (ax, feature_ax) = matplotlib.pyplot.subplots(figsize=figsize, nrows=2, **splt_kws)
    else:
        fig = ax.get_figure()

    ax = plt_audio.plot(legend=False, alpha=0.4, color='tab:grey', label='audio', ax=ax)
    ax.set_title(f"Min-ts={plt_min} || Max-ts={plt_max}\n\
    Labeled Regions: word_code={word_code}, word='{data_map['word_code_d'][word_code]}'\
    \nSpeaking N windows={len(t_speaking_ixes)}; Silence N windows={len(t_speaking_ixes)}")
    ax2 = ax.twinx()

    ax2.set_ylim(0.05, 1.1)
    # ax.axvline(silence_min_ix / pd.Timedelta(1,'s'))
    # (data_map['stim'].reindex(data_map['audio'].index).fillna(method='ffill').loc[plt_min: plt_max] > 0).astype(
    #    int).plot(ax=ax2, color='tab:blue', label='original stim')
    (data_map['stim'].resample('5ms').first().fillna(method='ffill').loc[plt_min: plt_max] > 0).astype(
        int).plot(ax=ax2, color='tab:blue', label='stim')

    silence_s.plot(ax=ax2, color='red', lw=4, label='silence')
    speaking_s.plot(ax=ax2, color='green', lw=4, label=f"speaking ")
    ax.legend()
    ax2.legend()

    if feature_ax is not None:
        feat_df = data_map[feature_key].loc[plt_min: plt_max]
        feat_df.plot(ax=feature_ax,
                     cmap='viridis', grid=True,
                     alpha=0.44, legend=False)
        feature_ax.set_title(f"Features\nplot shape={feat_df.shape}); window length={len(t_speaking_ixes[0])}")

    fig.tight_layout()
    return fig, ax


def plot_region_over_signal(signal_s, region_min, region_max,
                            padding_time=pd.Timedelta('1s'),
                            plot_signal=True,
                            ax=None, signal_plot_kwargs=None, region_plot_kwargs=None):
    def_signal_plot_kwargs = dict(color='tab:green', alpha=0.5)
    if isinstance(signal_plot_kwargs, dict):
        def_signal_plot_kwargs.update(signal_plot_kwargs)
    elif signal_plot_kwargs is not None:
        raise ValueError()

    signal_plot_kwargs = def_signal_plot_kwargs

    region_plot_kwargs = dict() if region_plot_kwargs is None else region_plot_kwargs

    plt_min = region_min - padding_time
    # print(f"{plt_min} = {region_min} - {padding_time}")

    plt_max = region_max + padding_time
    # print(f"{plt_max} = {region_max} + {padding_time}")

    signal_s = signal_s.loc[plt_min: plt_max]
    plt_ix = signal_s.index

    region_line_s = pd.Series(0, index=plt_ix)
    region_line_s.loc[region_min: region_max] = 1

    ax2 = ax
    if plot_signal:
        ax = signal_s.loc[plt_min:plt_max].plot(ax=ax, **signal_plot_kwargs)
        ax2 = ax.twinx()

    ax2 = region_line_s.loc[plt_min:plt_max].plot(ax=ax2, **region_plot_kwargs)

    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    return fig, ax, ax2


def plot_multi_region_over_signal(signal_s, region_min_max_tuples,  # region_min, region_max,
                            padding_time=pd.Timedelta('1s'),
                            plot_signal=True,
                            ax=None, signal_plot_kwargs=None, region_plot_kwargs=None):
    def_signal_plot_kwargs = dict(color='tab:green', alpha=0.5)
    if isinstance(signal_plot_kwargs, dict):
        def_signal_plot_kwargs.update(signal_plot_kwargs)
    elif signal_plot_kwargs is not None:
        raise ValueError()

    signal_plot_kwargs = def_signal_plot_kwargs

    region_plot_kwargs = dict() if region_plot_kwargs is None else region_plot_kwargs

    region_min = min(t_ for t_, _t, _, _ in region_min_max_tuples)
    region_max = max(_t for t_, _t, _, _ in region_min_max_tuples)

    plt_min = region_min - padding_time

    plt_max = region_max + padding_time

    signal_s = signal_s.loc[plt_min: plt_max]
    plt_ix = signal_s.index

    ax2 = ax
    if plot_signal:
        ax = signal_s.loc[plt_min:plt_max].plot(ax=ax, **signal_plot_kwargs)
        ax2 = ax.twinx()

    region_lines_l = list()
    for t_, _t, label, v in region_min_max_tuples:
        region_line_s = pd.Series(0, index=plt_ix, name=label)
        region_line_s.loc[t_: _t] = v
        region_lines_l.append(region_line_s)

    region_df = pd.concat(region_lines_l, axis=1)

    ax2 = region_df.plot(ax=ax2, **region_plot_kwargs)

    fig = ax.get_figure()
    fig.patch.set_facecolor('white')

    return fig, ax, ax2
from ecog_speech import datasets, experiments, result_parsing, utils
from ecog_speech import pipeline as feature_processing
from dataclasses import dataclass, field
from ecog_speech import visuals as viz
import pandas as pd
import matplotlib
import numpy as np
import os



def plot_ucsd_sentence_regions(data_map,
                               listen_cols=('listening_region_start_t', 'listening_region_stop_t'),
                               speaking_cols=('speaking_region_start_t', 'speaking_region_stop_t'),
                               mouthing_cols=('mouthing_region_start_t', 'mouthing_region_stop_t'),
                               imagining_cols=('imagining_region_start_t', 'imagining_region_stop_t')
                               ):
    import matplotlib
    from tqdm.auto import tqdm
    ds_audio = data_map['audio'].resample('0.001S').mean()
    word_df = data_map['word_start_stop_times']

    plt_sent_codes = list(sorted(word_df.stim_sentcode.unique()))
    n_sent_codes = len(plt_sent_codes)
    n_rows = int(np.ceil(n_sent_codes / 2))
    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=2, figsize=(28, n_rows * 2.5))
    axs = axs.reshape(-1)

    for i, sent_code in (enumerate(tqdm(plt_sent_codes))):
        ax = axs[i]
        plt_word_df = word_df[word_df.stim_sentcode.eq(sent_code)]

        region_tuples = [
            (plt_word_df[listen_cols[0]].min(), plt_word_df[listen_cols[1]].max(), 'listening_region', 0.9),
            (plt_word_df[speaking_cols[0]].min(), plt_word_df[speaking_cols[1]].max(), 'speaking_region', 0.85),
            (plt_word_df[mouthing_cols[0]].min(), plt_word_df[mouthing_cols[1]].max(), 'mouth_region', 0.8),
            (plt_word_df[imagining_cols[0]].min(), plt_word_df[imagining_cols[1]].max(), 'imagine_region', 0.75),

         #   (plt_word_df['listening_region_start_t'].min(), plt_word_df['listening_region_stop_t'].max(),
         #    'listening_region', 0.9),
         #   (plt_word_df['start_t'].min(), plt_word_df['stop_t'].max(), 'speaking_region', 0.85),
         #   (plt_word_df['mouth_region_start_t'].min(), plt_word_df['mouth_stop_t'].max(), 'mouth_region', 0.8),
         #   (plt_word_df['imagine_region_start_t'].min(), plt_word_df['imagine_stop_t'].max(), 'imagine_region', 0.75),
        ]


        viz.plot_multi_region_over_signal(signal_s=ds_audio, region_min_max_tuples=region_tuples,
                                region_plot_kwargs=dict(ls='--', alpha=0.7, lw=4, title=f"sentcode={sent_code}"), ax=ax)
    fig.tight_layout()
    return fig, axs


def plot_ucsd_word_regions2(data_map, start_stop_label_tuples_l=(('speaking', 'start_t', 'stop_t', 'word'),),
                            #include_listen=None, include_speaking=None, include_imagine=None, include_mouth=None,
                            legend_kws=None, code_col=None,
                            **region_plot_overrides):
    import matplotlib
    from tqdm.auto import tqdm
    legend_kws = dict(fontsize=6.5, loc='upper center') if legend_kws is None else legend_kws

    ds_audio = data_map['audio'].resample('0.001S').mean()
    word_df = data_map['word_start_stop_times']

    start_stop_label_tuples_l = np.array(start_stop_label_tuples_l).tolist()

    plt_sent_codes = list(sorted(word_df.stim_sentcode.unique()))
    n_sent_codes = len(plt_sent_codes)
    n_rows = int(np.ceil(n_sent_codes / 2))
    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=2, figsize=(27, n_rows * 3))
    axs = axs.reshape(-1)

    for i, sent_code in (enumerate(tqdm(plt_sent_codes, 'Processing by sentence'))):
        # Map the region of the sentence to the word regions - e.g. spoken vs. imagine
        sent_region_map = dict()
        ax = axs[i]
        plt_word_df = word_df[word_df.stim_sentcode.eq(sent_code) & (word_df.word.str.upper() == word_df.word)].copy()

        if code_col is not None:
            plt_word_df['word'] = plt_word_df['word'] + '(' + plt_word_df[code_col].astype(str) + ')'

        for i, (key, start_col, stop_col, label_col) in enumerate(start_stop_label_tuples_l):
            sent_region_map[key] = plt_word_df[[start_col, stop_col, label_col]].assign(
                **{'v': 1, label_col: plt_word_df[label_col] + '_' + key}
            ).values.tolist()


        #if include_speaking or ('word' in plt_word_df.columns and include_speaking is None):
        #    sent_region_map['speaking'] = plt_word_df[['start_t', 'stop_t', 'word']].assign(v=1).values.tolist()

        #if include_imagine or ('imagine_start_t' in plt_word_df.columns and include_imagine is None):
        #    sent_region_map['imagining'] = (plt_word_df.assign(word=plt_word_df.word + '_img', v=1)
        #                                    [['imagine_start_t', 'imagine_stop_t', 'word', 'v']]
        #                                    .values.tolist())
        #if include_mouth or ('mouth_start_t' in plt_word_df.columns and include_mouth is None):
        #    sent_region_map['mouthing'] = (plt_word_df.assign(word=plt_word_df.word + '_mth', v=1)
        #                                   [['mouth_start_t', 'mouth_stop_t', 'word', 'v']]
        #                                   .values.tolist())

        #if include_listen or ('listen_start_t' in plt_word_df.columns and include_listen is None):
        #    sent_region_map['listen'] = (plt_word_df.assign(word=plt_word_df.word + '_lis', v=1)
        #                                 [['listen_start_t', 'listen_stop_t', 'word', 'v']]
        #                                 .values.tolist())


        region_tuples = sum(sent_region_map.values(), list())  #speaking_regions_l + imagine_regions_l
        region_plot_kwargs = dict(style='--', alpha=0.9, lw=4,
                                  title=f"sentcode={sent_code}",
                                  cmap='tab20')
        region_plot_kwargs.update(region_plot_overrides)
        fig, ax, ax2 = viz.plot_multi_region_over_signal(signal_s=ds_audio, region_min_max_tuples=region_tuples,
                                                         region_plot_kwargs=region_plot_kwargs,
                                                         ax=ax)
        ax2.legend(ncol=len(sent_region_map), **legend_kws)

    fig.tight_layout()

    return fig, axs




def plot_ucsd_word_regions(data_map, include_listen=None, include_speaking=None,
                           include_imagine=None, include_mouth=None,
                           legend_kws=None, code_col=None,
                           **region_plot_overrides):
    import matplotlib
    from tqdm.auto import tqdm
    legend_kws = dict(fontsize=6.5, loc='upper center') if legend_kws is None else legend_kws

    ds_audio = data_map['audio'].resample('0.001S').mean()
    word_df = data_map['word_start_stop_times']
    plt_sent_codes = list(sorted(word_df.stim_sentcode.unique()))
    n_sent_codes = len(plt_sent_codes)
    n_rows = int(np.ceil(n_sent_codes / 2))
    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=2, figsize=(27, n_rows * 3))
    axs = axs.reshape(-1)

    for i, sent_code in (enumerate(tqdm(plt_sent_codes, 'Processing by sentence'))):
        # Map the region of the sentence to the word regions - e.g. spoken vs. imagine
        sent_region_map = dict()
        ax = axs[i]
        plt_word_df = word_df[word_df.stim_sentcode.eq(sent_code) & (word_df.word.str.upper() == word_df.word)].copy()

        if code_col is not None:
            plt_word_df['word'] = plt_word_df['word'] + '(' + plt_word_df[code_col].astype(str) + ')'

        if include_speaking or ('word' in plt_word_df.columns and include_speaking is None):
            sent_region_map['speaking'] = plt_word_df[['start_t', 'stop_t', 'word']].assign(v=1).values.tolist()

        if include_imagine or ('imagine_start_t' in plt_word_df.columns and include_imagine is None):
            sent_region_map['imagining'] = (plt_word_df.assign(word=plt_word_df.word + '_img', v=1)
                                            [['imagine_start_t', 'imagine_stop_t', 'word', 'v']]
                                            .values.tolist())
        if include_mouth or ('mouth_start_t' in plt_word_df.columns and include_mouth is None):
            sent_region_map['mouthing'] = (plt_word_df.assign(word=plt_word_df.word + '_mth', v=1)
                                           [['mouth_start_t', 'mouth_stop_t', 'word', 'v']]
                                           .values.tolist())

        if include_listen or ('listen_start_t' in plt_word_df.columns and include_listen is None):
            sent_region_map['listen'] = (plt_word_df.assign(word=plt_word_df.word + '_lis', v=1)
                                         [['listen_start_t', 'listen_stop_t', 'word', 'v']]
                                         .values.tolist())


        region_tuples = sum(sent_region_map.values(), list())  #speaking_regions_l + imagine_regions_l
        region_plot_kwargs = dict(style='--', alpha=0.9, lw=4,
                                  title=f"sentcode={sent_code}",
                                  cmap='tab20')
        region_plot_kwargs.update(region_plot_overrides)
        fig, ax, ax2 = viz.plot_multi_region_over_signal(signal_s=ds_audio, region_min_max_tuples=region_tuples,
                                                         region_plot_kwargs=region_plot_kwargs,
                                                         ax=ax)
        ax2.legend(ncol=len(sent_region_map), **legend_kws)

    fig.tight_layout()

    return fig, axs


def plot_grid_of_label_regions(data_map):
    import matplotlib
    from tqdm.auto import tqdm
    from matplotlib.dates import DateFormatter

    n_to_plt = len(data_map['sample_index_map'])# - 1  # skip 0
    n_cols = 5
    n_rows = int(np.ceil(n_to_plt / (n_cols)))

    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(35, 3.9 * n_rows))
    axs = axs.reshape(-1)
    plt_scale = data_map['audio'].abs().quantile(.999)

    #word_codes_to_plt = set(data_map['sample_index_map'].keys()) - set([0])
    word_codes_to_plt = set(data_map['sample_index_map'].keys())# - set([0])

    sample_ix = 0
    ds_audio = data_map['audio'].resample('0.001S').mean()
    for i, word_code in tqdm(enumerate(word_codes_to_plt)):
        _ax = axs[i]
        wrd_ix = data_map['sample_index_map'][word_code][sample_ix]

        fig, ax, ax2 = viz.plot_region_over_signal(ds_audio, wrd_ix.min(), wrd_ix.max(),
                                                    region_plot_kwargs=dict(ls='--'),
                                                    padding_time=pd.Timedelta('1.5s'), ax=_ax)

        code_min_t = min(ix.min() for ix in data_map['sample_index_map'][word_code])
        code_max_t = max(ix.max() for ix in data_map['sample_index_map'][word_code])
        try:
            fig, ax, _ = viz.plot_region_over_signal(ds_audio,
                                                     code_min_t,
                                                     code_max_t,
                                                      #data_map['sample_index_map'][word_code][0].min(),
                                                      #data_map['sample_index_map'][word_code][-1].max(),
                                                      padding_time=pd.Timedelta('1.5s'),
                                                      region_plot_kwargs=dict(ls='-', zorder=0), ax=ax2, plot_signal=False)
        except ValueError as e:
            print(e)
            raise

        # ax.set_ylim(-plt_scale, plt_scale)
        _ax.grid(True)
        _ax.set_title(f"Word code = {word_code}\nN Windows = {len(data_map['sample_index_map'][word_code])}")
        _ax.tick_params(axis='x', labelsize=7)
        if i == 0:
            _ax.legend(['audio'])
            ax2.legend(['first window', 'windowing region'])

    # fig.suptitle(f"Showing {n_to_plt} word codes {len(data_map['sample_index_map'][0])} total silent regions", fontsize=20, y=1.01)
    fig.tight_layout()
    return fig


def plot_grid_of_silent_regions(data_map):
    n_to_plt = 50
    # n_to_plt = len(data_map['sample_index_map']) - 1 # skip 0
    n_cols = 5
    n_rows = int(np.ceil(n_to_plt / (n_cols)))
    i_to_plt = np.random.choice(list(range(len(data_map['sample_index_map'][0]))), n_cols * n_rows, replace=False)
    ixes_to_plt = [data_map['sample_index_map'][0][_i] for _i in i_to_plt]
    # ixes_to_plt = [data_map['sample_index_map'][0][_i]
    #               for _i in np.random.randint(0, len(data_map['sample_index_map'][0]), n_cols*n_rows)]

    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(35, 3.9 * n_rows))
    axs = axs.reshape(-1)
    plt_scale = data_map['audio'].abs().quantile(.999)
    ds_audio = data_map['audio'].resample('0.001S').mean()
    for i, (ix_i, ix) in enumerate(zip(i_to_plt, ixes_to_plt)):
        ax = axs[i]
        viz.plot_region_over_signal(ds_audio, ix.min(), ix.max(),
                                     region_plot_kwargs=dict(ls='--'), ax=ax, )
        ax.set_ylim(-plt_scale, plt_scale)
        ax.set_title(f"Silent index {ix_i} of {len(data_map['sample_index_map'][0])}")
        ax.grid(True)

    fig.tight_layout()
    fig.suptitle(f"Showing {n_to_plt} samples of {len(data_map['sample_index_map'][0])} total silent regions",
                 fontsize=20, y=1.01)
    return fig


def plot_grid_of_index_by_key(data_map, sample_index_key):
    n_to_plt = 50
    # n_to_plt = len(data_map['sample_index_map']) - 1 # skip 0
    n_cols = 5
    n_rows = int(np.ceil(n_to_plt / (n_cols)))
    ixes = data_map['sample_index_map'][sample_index_key]
    n_ixes = len(ixes)
    ix_source = data_map.get('index_source_map', dict()).get(sample_index_key, 'no source in map')

    i_to_plt = np.random.choice(list(range(len(ixes))), n_cols * n_rows, replace=False)
    ixes_to_plt = [ixes[_i] for _i in i_to_plt]

    # ixes_to_plt = [data_map['sample_index_map'][0][_i]
    #               for _i in np.random.randint(0, len(data_map['sample_index_map'][0]), n_cols*n_rows)]

    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(35, 3.9 * n_rows))
    axs = axs.reshape(-1)
    plt_scale = data_map['audio'].abs().quantile(.999)
    ds_audio = data_map['audio'].resample('0.001S').mean()
    for i, (ix_i, ix) in enumerate(zip(i_to_plt, ixes_to_plt)):
        ax = axs[i]
        viz.plot_region_over_signal(ds_audio, ix.min(), ix.max(),
                                     region_plot_kwargs=dict(ls='--'), ax=ax, )
        ax.set_ylim(-plt_scale, plt_scale)
        ax.set_title(f"Silent index {ix_i} of {n_ixes}")
        ax.grid(True)

    fig.tight_layout()
    fig.suptitle(f"Showing {n_to_plt} samples of {n_ixes} for key = {sample_index_key} (source: {ix_source})",
                 fontsize=20, y=1.01)
    return fig


def plot_label_inspection_figures(data_map):
    output_fig_map = dict()

    wrd_code_len_s = pd.Series({wrd_cd: len(ixes)
                                for wrd_cd, ixes in data_map['sample_index_map'].items()}, name='n_ixes')
    n_speak_wins = len(wrd_code_len_s)
    hist_title = f'Hist of Word Codes\' N={n_speak_wins} label windows ' \
                 f'{len(wrd_code_len_s)} unique word codes'
    hist_title += f'\nLongest regions: {wrd_code_len_s.nlargest(5).to_dict()}'
    ax = wrd_code_len_s.drop(0).plot.hist(title=hist_title)
    ax.set_xlabel('N Windows in label region')
    ax.set_ylabel('N Word codes')
    #ax.set_xlim(0)
    fig = ax.get_figure()
    # fig.patches.
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    #output_fig_map['silent_region_grid'] = plot_grid_of_silent_regions(data_map)
    #output_fig_map['label_region_grid'] = plot_grid_of_label_regions(data_map)
    output_fig_map['word_code_win_histo'] = fig
    output_fig_map.update(**{f'sample_index_key_{k}': plot_grid_of_index_by_key(data_map, k)
                             for k in data_map['sample_index_map'].keys()})

    return output_fig_map


@dataclass
class PlotLabelRegionOptions:
    output_dir: str = './'
    dataset_name: str = 'hvs'
    data_subset: str = 'UCSD'
    pre_proc_pipeline: str = 'word_level'
    n_workers: int = 2


if __name__ == """__main__""":
    from simple_parsing import ArgumentParser
    from tqdm.auto import tqdm
    from matplotlib import pyplot as plt
    import pathlib

    parser = ArgumentParser()
    parser.add_arguments(PlotLabelRegionOptions, dest='plt_label_region_opts')
    args = parser.parse_args()
    plt_label_region_opts: PlotLabelRegionOptions = args.plt_label_region_opts

    psubset = plt_label_region_opts.data_subset
    dataset_cls = datasets.BaseDataset.get_dataset_by_name(plt_label_region_opts.dataset_name)
    pre_proc_pipeline = plt_label_region_opts.pre_proc_pipeline

    force = True
    def run_one(pid, pt, ptuples):
        output_path = f'{psubset}-{pt[1]}-{pt[3]}_label_inspection_plots.pdf'
        output_path = os.path.join(plt_label_region_opts.output_dir, output_path)
        if pathlib.Path(output_path).is_file() and not force:
            print("Skipping " + str(output_path))
            return
        print("LOADING NWW")
        dset = dataset_cls(patient_tuples=[pt], pre_processing_pipeline=pre_proc_pipeline)
        print("Getting data map")
        data_map = dset.data_maps[pt]
        print("Plotting")
        fig_map = plot_label_inspection_figures(data_map)

        if psubset == 'UCSD':# and pre_proc_pipeline == 'audio_gate_imagine':
            fig, _ = plot_ucsd_word_regions2(data_map,# include_listen=False, include_speaking=True,
                                            #include_imagine=False, include_mouth=False
                                             )
            fig_map['UCSD_words'] = fig

            fig, _ = plot_ucsd_sentence_regions(data_map)
            fig_map['UCSD_sentence'] = fig

        from matplotlib.backends.backend_pdf import PdfPages

        print("saving plots")
        # create a PdfPages object
        pdf = PdfPages(output_path)

        for fig_name, _fig in fig_map.items():
            pdf.savefig(_fig, bbox_inches='tight')
        plt.close('all')
        pdf.close()
        del dset
        del pdf
        print("All done")


    from multiprocessing import Pool
    p = Pool(plt_label_region_opts.n_workers) if plt_label_region_opts.n_workers > 1 else None
    for pid, ptuples in tqdm(dataset_cls.all_patient_maps[psubset].items()):
        for pt in ptuples:
            if p is not None:
                p.apply_async(run_one, kwds=dict(pid=pid, pt=pt, ptuples=ptuples))
            else:
                run_one(pid=pid, pt=pt, ptuples=ptuples)

    if p is not None:
        p.close()
        p.join()

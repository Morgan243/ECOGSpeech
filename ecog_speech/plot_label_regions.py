from ecog_speech import datasets, experiments, result_parsing, utils
from ecog_speech import pipeline as feature_processing
from ecog_speech import visuals as viz
import pandas as pd
import matplotlib
import numpy as np


from scipy.io import loadmat

# - For Set of patients
# - Plot all of their word labels
# - Plot histogram of word code label n samples
# - Plot sample of silent regions
#     - Silence is not contigoous, so harder to show all windpws

#pl = (feature_processing.SubsampleECOG()
#     >> feature_processing.PowerThreshold(window_samples=48000 // 4)
#     >> feature_processing.SampleIndicesFromStimV2(silent_window_scale=5))


datasets.NorthwesternWords.default_base_path = '/export/datasets/ASPEN/ecog/Data2Lore/SingleWord/'


def plot_grid_of_label_regions(data_map):
    import matplotlib
    from tqdm.auto import tqdm
    from matplotlib.dates import DateFormatter

    n_to_plt = len(data_map['sample_index_map']) - 1  # skip 0
    n_cols = 5
    n_rows = int(np.ceil(n_to_plt / (n_cols)))

    fig, axs = matplotlib.pyplot.subplots(nrows=n_rows, ncols=n_cols, figsize=(35, 3.9 * n_rows))
    axs = axs.reshape(-1)
    plt_scale = data_map['audio'].abs().quantile(.999)

    word_codes_to_plt = set(data_map['sample_index_map'].keys()) - set([0])

    sample_ix = 0
    for i, word_code in tqdm(enumerate(word_codes_to_plt)):
        _ax = axs[i]
        wrd_ix = data_map['sample_index_map'][word_code][sample_ix]

        fig, ax, ax2 = viz.plot_region_over_signal(data_map['audio'], wrd_ix.min(), wrd_ix.max(),
                                                    region_plot_kwargs=dict(ls='--'),
                                                    padding_time=pd.Timedelta('1.5s'), ax=_ax)
        fig, ax, _ = viz.plot_region_over_signal(data_map['audio'],
                                                  data_map['sample_index_map'][word_code][0].min(),
                                                  data_map['sample_index_map'][word_code][-1].max(),
                                                  padding_time=pd.Timedelta('1.5s'),
                                                  region_plot_kwargs=dict(ls='-', zorder=0), ax=ax2, plot_signal=False)

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
    for i, (ix_i, ix) in enumerate(zip(i_to_plt, ixes_to_plt)):
        ax = axs[i]
        viz.plot_region_over_signal(data_map['audio'], ix.min(), ix.max(),
                                     region_plot_kwargs=dict(ls='--'), ax=ax, )
        ax.set_ylim(-plt_scale, plt_scale)
        ax.set_title(f"Silent index {ix_i} of {len(data_map['sample_index_map'][0])}")
        ax.grid(True)

    fig.tight_layout()
    fig.suptitle(f"Showing {n_to_plt} samples of {len(data_map['sample_index_map'][0])} total silent regions",
                 fontsize=20, y=1.01)
    return fig


def plot_label_inspection_figures(data_map):
    output_fig_map = dict()

    wrd_code_len_s = pd.Series({wrd_cd: len(ixes)
                                for wrd_cd, ixes in data_map['sample_index_map'].items()}, name='n_ixes')
    n_speak_wins = len(wrd_code_len_s.drop(0))
    hist_title = f'Hist of Word Codes\' N={n_speak_wins} label windows\nN={wrd_code_len_s.loc[0]} silence samples (not inc. histo)\n{len(wrd_code_len_s)} unique word codes'
    hist_title += f'\nLongest regions: {wrd_code_len_s.drop(0).nlargest(5).index.tolist()}'
    ax = wrd_code_len_s.drop(0).plot.hist(title=hist_title)
    ax.set_xlabel('N Windows in label region')
    ax.set_ylabel('N Word codes')
    ax.set_xlim(0)
    fig = ax.get_figure()
    # fig.patches.
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    output_fig_map['silent_region_grid'] = plot_grid_of_silent_regions(data_map)
    output_fig_map['label_region_grid'] = plot_grid_of_label_regions(data_map)

    output_fig_map['word_code_win_histo'] = fig


    return output_fig_map


if __name__ == """__main__""":
    from tqdm.auto import tqdm
    from matplotlib import pyplot as plt
    from sklearn.pipeline import FeatureUnion, Pipeline
    import pathlib
    psubset = 'MC'

    sk_pl = Pipeline([
        ('subsample', feature_processing.SubsampleSignal()),
        ('Threshold', feature_processing.PowerThreshold(speaking_window_samples=48000 // 16,
                                                        silence_window_samples=int(48000 * 1.5),
                                                        speaking_quantile_threshold=0.9,
                                                            #silence_threshold=0.001,
                                                        #silGence_quantile_threshold=0.05,
                                                        silence_n_smallest=5000,
                                                           )),
        ('speaking_indices', feature_processing.WindowSampleIndicesFromStim('stim_pwrt',
                                                                            target_onset_shift=pd.Timedelta(-.5, 's'),
                                                                            # input are centers, and output is a window of .5 sec
                                                                            # so to center it, move the point (center) back .25 secods
                                                                            # so that extracted 0.5 sec window saddles the original center
                                                                            #target_offset_shift=pd.Timedelta(-0.25, 's')
                                                                            target_offset_shift=pd.Timedelta(-0.5, 's')
                                                                            )
         ),

        ('silence_indices', feature_processing.WindowSampleIndicesFromIndex('silence_stim_pwrt_s',
                                                                            # Center the extracted 0.5 second window
                                                                            index_shift=pd.Timedelta(-0.25, 's'),
                                                                            stim_value_remap=0
                                                                          )),
        ('output', 'passthrough')
    ])

    def run_one(pid, pt, ptuples):
        output_path = f'{psubset}-{pt[1]}-{pt[3]}_label_inspection_plots.pdf'
        if pathlib.Path(output_path).is_file():
            print("Skipping " + str(output_path))
            return
        print("LOADING NWW")
        dset = datasets.NorthwesternWords(patient_tuples=[pt], pre_processing_pipeline=sk_pl.transform)
        print("Getting data map")
        data_map = dset.data_maps[pt]
        # Should only be onw
        # for i, (pt, data_map) in enumerate(dset.data_maps.items()):
        print("Plotting")
        fig_map = plot_label_inspection_figures(data_map)

        from matplotlib.backends.backend_pdf import PdfPages

        print("saving plots")
        # create a PdfPages object
        pdf = PdfPages(output_path)

        for fig_name, _fig in fig_map.items():
            pdf.savefig(_fig)
        plt.close('all')
        pdf.close()
        del dset
        del pdf
        print("All done")


    from multiprocessing import Pool
    p = Pool(5)
    for pid, ptuples in tqdm(datasets.NorthwesternWords.all_patient_maps[psubset].items()):
        for pt in ptuples:
            p.apply_async(run_one, kwds=dict(pid=pid, pt=pt, ptuples=ptuples))

    p.close()
    p.join()

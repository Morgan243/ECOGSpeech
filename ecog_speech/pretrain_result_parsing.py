from dataclasses import dataclass
from simple_parsing.helpers import JsonSerializable
from typing import Optional
#from simple_parsing.utils import Json

import pandas as pd
import numpy as np
from ecog_speech.experiments import transfer_learning
from ecog_speech import datasets


def pca_and_scatter_plot(_df):
    from sklearn.decomposition import PCA

    _pca_arr = PCA(2).fit_transform(_df)

    _pca_df = pd.DataFrame(_pca_arr)

    ax = _pca_df.plot.scatter(x=0, y=1, alpha=0.3, c=all_y_s, cmap='tab10', sharex=False)
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    return _pca_df, fig, ax


def pca_and_pair_plot(_df, _y_s=None, n_components=3, **pair_plt_kws):
    from sklearn.decomposition import PCA

    _pca_arr = PCA(n_components).fit_transform(_df)

    _pca_df = pd.DataFrame(_pca_arr)

    import seaborn as sns

    plt_kws = dict(  # hue='target_val',
        diag_kws=dict(common_norm=False), kind='hist',
        diag_kind='kde', )
    _plt_df = _pca_df
    if _y_s is not None:
        _plt_df = _plt_df.join(_y_s)
        plt_kws['hue'] = _y_s.name

    plt_kws.update(pair_plt_kws)
    g = sns.pairplot(_plt_df, **plt_kws)

    # ax = _pca_df.plot.scatter(x=0, y=1, alpha=0.3, c=all_y_s, cmap='tab10', sharex=False)
    #fig = ax.get_figure()
    #fig.patch.set_facecolor('white')
    return _pca_df, g.figure, g


@dataclass
class PretrainedResultParsing(JsonSerializable):
    result_file: str = None
    print_results: bool = False
    base_model_path: Optional[str] = None
    eval_sets: Optional[str] = None

    def run(self):
        pretrained_model, result_json = transfer_learning.FineTuningExperiment.load_pretrained_model_results(self.result_file)
        pretrain_sets = result_json['dataset_options']['train_sets']

        target_tuples = datasets.HarvardSentences.make_remaining_tuples_from_selected(pretrain_sets)
        #pretrain_sets, target_tuples
        loss_df = pd.DataFrame(result_json['epoch_outputs']).T
        #loss_df

        loss_df[['total_loss', 'cv_loss', 'accuracy']].plot(secondary_y='accuracy', logy=True)

        result_json['model_options']

        cog2vec = pretrained_model

        hvs_test = datasets.HarvardSentences(target_tuples,
                                             # pre_processing_pipeline='audio_gate',
                                             pre_processing_pipeline='region_classification',
                                             flatten_sensors_to_samples=True,
                                             extra_output_keys='sensor_ras_coord_arr'
                                             )

        hvs_test = datasets.HarvardSentences(target_tuples,
                                             pre_processing_pipeline='audio_gate',
                                             # pre_processing_pipeline='region_classification',
                                             flatten_sensors_to_samples=True, extra_output_keys='sensor_ras_coord_arr'
                                             )

        class_val_to_label_d = hvs_test.data_maps[target_tuples[0]]['index_source_map']

        device = 'cuda'

        from tqdm.auto import tqdm
        import torch

        results_l = list()

        # cog2vec.eval()
        m = cog2vec.to(device).eval()
        # _dl = cv_dl
        _dl = hvs_test.to_dataloader(num_workers=4, batch_size=256, batches_per_epoch=100)
        # sens_id = 10
        for batch_d in tqdm(_dl, desc="Batching"):
            # X_barr = batch_d['signal_arr'].to(trainer.device)
            # Select a single sensor for now and remove the singleton dimension
            # sens_id = np.random.randint(0, X_barr.shape[1])
            # X = X_barr.select(1, sens_id).unsqueeze(1)

            with torch.no_grad():
                dev_batch_d = {k: arr.to(device) for k, arr in batch_d.items()}
                feat_d = m.forward(dev_batch_d, features_only=True, mask=False)
                results_l.append(
                    dict(signal_arr=batch_d['signal_arr'],
                         target_arr=batch_d['target_arr'],
                         # signal_arr=batch_d['signal_arr'].detach().cpu().numpy(),
                         **{n: arr.detach().cpu().numpy() for n, arr in feat_d.items()}))

        all_x = np.concatenate([r['x'] for r in results_l])
        all_x_df = pd.DataFrame(all_x.reshape(all_x.shape[0], -1))

        all_feats = np.concatenate([r['features'] for r in results_l])
        all_feats_df = pd.DataFrame(all_feats.reshape(all_feats.shape[0], -1))

        # If there was a target
        all_y = np.concatenate([r['target_arr'] for r in results_l])
        all_y_s = pd.Series(all_y.squeeze(), name='target_val')

        all_x_df.describe()

        corr_speak_df = all_x_df.corrwith(all_y_s)
        ax = corr_speak_df.hist()


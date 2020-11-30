import pandas as pd
from glob import glob
from os import path
import numpy as np
import torch
from torch.utils import data as tdata
from tqdm.auto import tqdm
import h5py

from BCI2kReader import BCI2kReader as b2k
from os import environ
import os
import attr
import torchaudio
import socket
from ecog_speech import feature_processing

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')


## TODO
# - Datasets
#   - Open ECOG from stanford (a bunch of tasks, including speech)
#   - DEAP dataset for emotion recognition
#   - VR + Workload (ball into a cup in VR) wearing EEG

class BaseDataset(tdata.Dataset):
    def to_dataloader(self, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, **kwargs):
        dset = self
        if random_sample:
            if batches_per_epoch is None:
                batches_per_epoch = len(dset) // batch_size

            dataloader = tdata.DataLoader(dset, batch_size=batch_size,
                                          sampler=tdata.RandomSampler(dset,
                                                                      replacement=True,
                                                                      num_samples=batches_per_epoch * batch_size),
                                          shuffle=shuffle, num_workers=num_workers,
                                          **kwargs)
        else:
            dataloader = tdata.DataLoader(dset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers,
                                          **kwargs)
        return dataloader

    @classmethod
    def with_env_key(cls, env_key):
        pass


@attr.s
class RollDimension:
    roll_dim = attr.ib(1)
    min_roll = attr.ib(-2)
    max_roll = attr.ib(2)
    return_roll_amount = attr.ib(False)

    def __call__(self, sample):
        roll_amount = int(np.random.random_integers(self.min_roll,
                                                    self.max_roll))
        #return torch.roll(sample, roll_amount, self.roll_dim)
        r_sample = np.roll(sample, roll_amount, self.roll_dim)

        if self.return_roll_amount:
            ret = r_sample, roll_amount
        else:
            ret = r_sample

        return ret

@attr.s
class DEAP(BaseDataset):
    env_key = "DEAP_DATASET"
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 pkg_data_dir))
    base_path = attr.ib(None)

    eeg_sensor_ixes = list(range(1, 33))

    channel_name_map = {1: 'Fp1',
                        2: 'AF3',
                        3: 'F3',
                        4: 'F7',
                        5: 'FC5',
                        6: 'FC1',
                        7: 'C3',
                        8: 'T7',
                        9: 'CP5',
                        10: 'CP1',
                        11: 'P3',
                        12: 'P7',
                        13: 'PO3',
                        14: 'O1',
                        15: 'Oz',
                        16: 'Pz',
                        17: 'Fp2',
                        18: 'AF4',
                        19: 'Fz',
                        20: 'F4',
                        21: 'F8',
                        22: 'FC6',
                        23: 'FC2',
                        24: 'Cz',
                        25: 'C4',
                        26: 'T8',
                        27: 'CP6',
                        28: 'CP2',
                        29: 'P4',
                        30: 'P8',
                        31: 'PO4',
                        32: 'O2',

                        ####
                        33: 'hEOG (horizontal EOG, hEOG1 - hEOG2)',
                        34: 'vEOG (vertical EOG, vEOG1 - vEOG2)',
                        35: 'zEMG (Zygomaticus Major EMG, zEMG1 - zEMG2)',
                        36: 'tEMG (Trapezius EMG, tEMG1 - tEMG2)',
                        37: 'GSR (values from Twente converted to Geneva format (Ohm))',
                        38: 'Respiration belt',
                        39: 'Plethysmograph',
                        40: 'Temperature'}

    def __attrs_post_init__(self):
        self.__update_paths__()

    def __update_paths__(self):
        self.subject_path_map = {int(path.split(p)[-1].replace('.dat', '')[1:]): p
                                 for p in glob(path.join(self.base_path, '*.dat'))}
        self.subject_path_map_npz = {int(path.split(p)[-1].replace('.npz', '')[1:]): p
                                     for p in glob(path.join(self.base_path, '*.npz'))}

    @classmethod
    def load_dat_from_path(cls, path):
        import _pickle as cPickle
        return cPickle.load(open(path, 'rb'), encoding='bytes')

    @classmethod
    def load_all_from_dat(cls, subject_path_map):
        return {sid.decode('utf-8'): cls.load_dat_from_path(p)
                for sid, p in tqdm(subject_path_map.items(), desc='reading dat')}
        # return subject_data_map

    @classmethod
    def save_to_npz(cls, subject_data_map, base_path):
        for sid, dmap in tqdm(subject_data_map.items()):
            np.savez(path.join(base_path, f's{sid}.npz'),
                     **{k: arr for k, arr in dmap.items()})

    @classmethod
    def load_all_from_npz(cls, subject_path_map):
        return {sid: np.load(p)
                for sid, p in tqdm(subject_path_map.items(), desc='reading npz')}

    def maybe_convert_to_npz(self):
        n_dat_files = len(self.subject_path_map)
        n_npz_files = len(self.subject_path_map_npz)

        if n_dat_files < n_npz_files:
            msg = f"Somehow there are more .dat files than .npz in {self.base_path}"
            raise ValueError(msg)
        elif n_dat_files == n_npz_files:
            print("Same number of dat and npz files - no need to convert")
        else:
            subject_p_map = {sid: p for sid, p in self.subject_path_map.items()
                             if sid not in self.subject_path_map_npz}
            dat_map = self.load_all_from_dat(subject_p_map)
            self.save_to_npz(dat_map, self.base_path)


@attr.s
class HarvardSentences(BaseDataset):
    """
    <TODO Dataset info>
    Expects the Formatted Datasets folder (TODO - vetted?)
    Uses filenames for metadata and partition datasets before loading
    """
    env_key = 'HARVARDSENTENCES_DATASET'
    default_hvs_path = path.join(pkg_data_dir, '2-Formatted Datasets/2-Formatted Datasets/Harvard Sentences/')
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 default_hvs_path))
    base_path = attr.ib(None)
    window_description = attr.make_class("HarvardSentenceWindowDesc",
                                         dict(  # dataset_key=attr.ib(),
                                             lab_name=attr.ib(), sid=attr.ib(), tid=attr.ib(),
                                             ecog_i=attr.ib(),
                                             ecog_width=attr.ib(),
                                             wav_i=attr.ib(),
                                             wav_width=attr.ib(),
                                             label_id=attr.ib(),
                                         ))

    @classmethod
    def get_data_paths(cls, base_path=None, lab_name=None):
        base_path = cls.default_base_path if base_path is None else base_path
        lab_name = '*' if lab_name is None else lab_name
        return glob(os.path.join(base_path, lab_name, 'Data', '*.mat'))

    @classmethod
    def load_mat_keys_from_path(cls, p):
        """
        Returns only the keys in the HDF5 file without loading the data
        """
        with h5py.File(p, 'r') as f:
            keys = list(f.keys())
        return keys

    @classmethod
    def load_mat_from_path(cls, p):
        """
        Loads all keys in HDF5 file into dict, convert values to np.array
        """
        with h5py.File(p, 'r') as f:
            mat_dat_map = {k: np.array(f[k]) for k in f.keys()}
        return mat_dat_map

    @classmethod
    def parse_meta_fname(cls, fname):
        """
        Given fname, return its parts as <lab><sid>{_Task_}<tid>.mat
        TODO: Some hacky conventions/implementation to reconsider
        """
        f = fname.replace('.mat', '')
        if f[:3] == 'VCU':
            lab = 'VCU'
            try:
                sid, _, tid = f[3:].split('_')
            except:
                sid, tid = f[3:], '00'

        if f[:4] == 'UCSD':
            lab = 'UCSD'
            sid, _, tid = f[4:].split('_')

        return lab, sid, tid

    @classmethod
    def load(cls, base_path=None, lab_name=None,
             subject_id=None, task_id=None,
             load_f=None, path_as_key=False):
        """
        general purpose load method that maps the load_f onto the datasets
        filtered to and located by the other parameters (i.e. lab_name, subject_id,
        task_id).

        The key of the returned dictionary will be a tuple of metadata or the full string
        path (path_as_key=True).
        """
        load_f = cls.load_mat_from_path if load_f is None else load_f
        paths = cls.get_data_paths(base_path, lab_name)
        # Pull meta data from filenames
        path_to_meta = {p: cls.parse_meta_fname(os.path.split(p)[-1])
                        for p in paths}

        # Simple filtering scheme
        to_load = {p: m for p, m in path_to_meta.items()
                   if all((lab_name is None or lab_name == m[0],
                           subject_id is None or subject_id == m[1],
                           task_id is None or task_id == m[2]
                           ))}

        return {p if path_as_key else m: load_f(p)
                for p, m in tqdm(to_load.items())}

    @classmethod
    def load_meta_frame(cls, base_path=None, lab_name=None):
        """
        Return a frame containing indicators for keys found in each of the datasets
        """
        hvs_keys_map = cls.load(base_path=base_path, lab_name=lab_name,
                                load_f=cls.load_mat_keys_from_path,
                                path_as_key=True)

        _iter = zip(hvs_keys_map.items(), [cls.parse_meta_fname(os.path.split(p)[-1]) for p in hvs_keys_map])
        hvs_meta_df = pd.DataFrame([dict(lab=m[0], sid=m[1], tid=m[2], path=p,
                                         **{v: 1 for v in l})
                                    for (p, l), m in _iter])
        return hvs_meta_df

    @classmethod
    def make_window_desc(cls, d_map=None,
                         ecog_width=350, ecog_step=25,

                         base_path=None, lab_name=None,
                         subject_id=None, task_id=None):
        if d_map is None:
            d_map = cls.load(base_path=base_path, lab_name=lab_name,
                             subject_id=subject_id, task_id=task_id)

        for md, mat_d in d_map.items():
            for ecog_i in range(0, mat_d['sEEG'].shape[-1], ecog_step):
                desc = cls.window_description(*md, ecog_i, ecog_width=ecog_width,
                                              wav_i=None, wave_width=None, label_id=None)


# Currently this class handles *multiple* patients' data
# Thinking this will make it more flexable later

@attr.s
class NorthwesternWords(BaseDataset):
    env_key = 'NORTHWESTERNWORDS_DATASET'
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 path.join(pkg_data_dir,
                                                           '3-Vetted Datasets',
                                                           'Single Word')
                                                 ))
    base_path = attr.ib(None)
    # TODO: named tuple
    patient_tuples = attr.ib((

                              ('Mayo Clinic', 19, 1, 1),
                              ('Mayo Clinic', 19, 1, 2),
                              ('Mayo Clinic', 19, 1, 3),

                              #('Mayo Clinic', 21, 1, 1),
                              #('Mayo Clinic', 21, 1, 2),

         #('Mayo Clinic', 22, 1, 1),
         #('Mayo Clinic', 22, 1, 2),
         #('Mayo Clinic', 22, 1, 3),
         #('Mayo Clinic', 22, 1, 4),

    ))
    # num_mfcc = 13
    sensor_columns = list(range(64))
    audio_sample_rate = 48000
    ecog_sample_rate = 1200
    ecog_pass_band = attr.ib((70, 250))

    # Hack
    ecog_window_shift_sec = pd.Timedelta(0.5, 's')
    #ecog_window_step_sec = attr.ib(0.01, factory=lambda s: pd.Timedelta(s, 's'))
    ecog_window_step_sec = pd.Timedelta(0.01, 's')
    ecog_window_n = attr.ib(50)

    # In terms of audio samples
    # fixed_window_size = attr.ib(audio_sample_rate * 1)
    # fixed_window_step = attr.ib(int(audio_sample_rate * .01))
    max_ecog_window_size = attr.ib(600)
    num_mfcc = attr.ib(13)
    verbose = attr.ib(True)

    # TODO: make the fields mappable from a dictionary of functions
    # Trying this out: define a small data schema class
    schema_fields = ["ecog", "ecog_arr",
                     "speech", "speech_arr",
                     # "mfcc",
                     "mfcc_arr",
                     "text", "text_arr"]
    sample_schema = attr.make_class("NorthwesterWordsSample",
                                    schema_fields)
    sample_schema2 = attr.make_class("NorthwesterWordsSample2",
                                     {f: attr.ib(None) for f in schema_fields})
    selected_word_indices = attr.ib(None)

    stim_indexing_source = attr.ib('stim_diff')
    transform = attr.ib(None)

    power_threshold = attr.ib(0.007)
    power_q = attr.ib(0.5)
    pre_processing_pipeline = attr.ib(None)
    data_from: 'NorthwesternWords' = attr.ib(None)

    def _make_pipeline_map(self):
        cls = self.__class__

        p_map = {
            'threshold': [
                (cls.default_preprocessing, dict()),
                (cls.stim_adjust__power_threshold, dict(threshold=self.power_threshold)),
            ],
            'quantile': [
                (cls.default_preprocessing, dict()),
                (cls.stim_adjust__power_quantile, dict(q=self.power_q)),
            ],
            'minimal': [
                (cls.default_preprocessing, dict()),
            ]
        }
        p_map['default'] = p_map['quantile']
        return p_map

    def __attrs_post_init__(self):
        self.pipeline_map = self._make_pipeline_map()
        if self.pre_processing_pipeline is None:
            self.pipeline_steps = self.pipeline_map['default']
        elif isinstance(self.pre_processing_pipeline, str):
            self.pipeline_steps = self.pipeline_map[self.pre_processing_pipeline]
        else:
            self.pipeline_steps = self.pre_processing_pipeline

        if self.data_from is None:
            self.mfcc_m = torchaudio.transforms.MFCC(self.audio_sample_rate,
                                                     self.num_mfcc)

            ## Data loading
            ## Load the data, parsing into pandas data frame/series types
            data_iter = tqdm(self.patient_tuples, desc="Loading data")
            self.data_map = {l_p_s_t_tuple:
                                 self.load_data(*l_p_s_t_tuple, verbose=self.verbose)
                             for l_p_s_t_tuple in data_iter}

            ## Data pre-processing
            # Cleanup the raw data at the global level
            for l_p_s_t_tuple in self.data_map.keys():
                frame_d = self.data_map[l_p_s_t_tuple]
                for p_i, (p_func, p_kws) in enumerate(self.pipeline_steps):
                    print(f"[{p_i}] {p_func.__name__}({','.join(p_kws.keys())})")
                    updates, remaps = p_func(frame_d, **p_kws)
                    in_use_kw = [nk for nk in updates.keys() if nk in frame_d]
                    if len(in_use_kw) > 0:
                        msg = ("Processing KW already in use: %s" %(", ".join(in_use_kw)))
                        raise ValueError(msg)

                    msg = "\t->Outputs: " + ", ".join(updates.keys())
                    print(msg)
                    frame_d.update(updates)
                    for src_name, new_mapping in remaps.items():
                        frame_d['remap'][src_name] = new_mapping
                self.data_map[l_p_s_t_tuple] = frame_d


            ## Sample Indexing
            # Produce a map from label mask to window indices
            self.sample_index_maps = {k_t: self.make_sample_indices(frame_d[frame_d['remap'][self.stim_indexing_source]],
                                                                    self.max_ecog_window_size,
                                                                    self.ecog_window_n)
                                      for k_t, frame_d in tqdm(self.data_map.items(),
                                                               desc="Extracting windows from "
                                                                    + self.stim_indexing_source)}
            # Creat a flat mapping for fully defined key
            self.flat_index_map = {tuple([wrd_id, ix_i] + list(k_t)): ixes
                                   for k_t, index_map in self.sample_index_maps.items()
                                       for wrd_id, ix_list in index_map.items()
                                           for ix_i, ixes in enumerate(ix_list)}

            #self.i_to_keys = {i: (k, k[2:]) for i, k in enumerate(self.flat_index_map.keys())}
            # Enumerate all the keys across flat_index_map into one large list for index-style,
            # has a len() and can be indexed into nicely (via np.ndarray)
            self.flat_keys = np.array([(k, k[2:])
                                       for i, k in enumerate(self.flat_index_map.keys())],
                                      dtype='object')
        else:
            print("Warning: using naive cross-referencing")
            self.mfcc_m = self.data_from.mfcc_m
            self.data_map = self.data_from.data_map
            self.sample_index_maps = self.data_from.sample_index_maps
            self.flat_index_map = self.data_from.flat_index_map
            self.flat_keys = self.data_from.flat_keys

        if self.selected_word_indices is not None:
            self.selected_flat_keys = self.flat_keys[self.selected_word_indices]
        else:
            self.selected_flat_keys = self.flat_keys

    def __len__(self):
        return len(self.selected_flat_keys)

    def __getitem__(self, item):
        ix_k, data_k = self.selected_flat_keys[item]
        data_d = self.data_map[data_k]

        so = self.make_sample_object(self.flat_index_map[ix_k],
                                     ix_k[0], data_d['ecog'],
                                     data_d['audio'],
                                     ecog_transform=self.transform,
                                     #max_ecog_samples=self.max_ecog_window_size,
                                     mfcc_f=self.mfcc_m)

        #if self.transform is not None:
        #    so.ecog_arr = self.transform(so.ecog_arr)

        return {k: v for k, v in attr.asdict(so).items()
                if isinstance(v, torch.Tensor)}

    def sample_plot(self, i, band=None,
                    offset_seconds=0,
                    figsize=(15, 10), axs=None):
        import matplotlib
        from IPython.display import display
        # offs = pd.Timedelta(offset_seconds)
        #t_word_ix = self.word_index[self.word_index == i].index
        ix_k, data_k = self.selected_flat_keys[i]
        t_word_ix = self.flat_index_map[ix_k]
        offs_td = pd.Timedelta(offset_seconds, 's')
        t_word_slice = slice(t_word_ix.min() - offs_td, t_word_ix.max() - offs_td)
        display(t_word_slice)
        display(t_word_ix.min() - offs_td)
        # t_word_ix = self.word_index.loc[t_word_ix.min() - offs_td: t_word_ix.max() - offs_td].index
        # t_word_ecog_df = self.ecog_df.reindex(t_word_ix).dropna()
        # t_word_wav_df = self.speech_df.reindex(t_word_ix)
        ecog_df = self.data_map[data_k]['ecog']
        speech_df = self.data_map[data_k]['audio']
        word_txt = self.data_map[data_k]['word_code_d'].get(ix_k[0])

        t_word_ecog_df = ecog_df.loc[t_word_slice].dropna()
        t_word_wav_df = speech_df.loc[t_word_slice]
        #display(t_word_ecog_df.describe())
        scols = self.sensor_columns

        ecog_std = ecog_df[scols].std()
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=ecog_std.min(), vmax=ecog_std.max())
        c_f = lambda v: cmap(norm(v))
        colors = ecog_std.map(c_f).values

        if axs is None:
            fig, axs = matplotlib.pyplot.subplots(figsize=figsize, nrows=2)
        else:
            fig = axs[0].get_figure()

        if band is not None:
            plt_df = t_word_ecog_df[scols].pipe(feature_processing.filter, band=band,
                                                sfreq=self.ecog_sample_rate)
        else:
            plt_df = t_word_ecog_df[scols]

        ax = plt_df.plot(alpha=0.3, legend=False,
                         color=colors, lw=1.2,
                         ax=axs[0], fontsize=14)
        ax.set_title(f"{len(plt_df)} samples")

        ax = t_word_wav_df.plot(alpha=0.7, legend=False, fontsize=14, ax=axs[1])
        ax.set_title(f"{len(t_word_wav_df)} samples, word = {word_txt}")

        fig.tight_layout()

        return axs

    @classmethod
    def make_sample_indices(cls, label_index, win_size, ecog_window_n=30):
        """
        Group by a label index (repeated label measures, one per sample)
        to find start and end points, then slice at the appropriate offsets
        back into the original data to capture the index (not the data).


        :param label_index:
        :param win_size:
        :param win_step:
        :return:
        """
        sample_indices = dict()
        # Repeated labels, group
        grp = label_index.groupby(label_index)
        for wrd_id, wave_wrd_values in grp:
            ixes = wave_wrd_values.index

            # ixes = label_index.index
            # if win_size > len(ixes):
            start_t = ixes.min()
            if wrd_id > 0:
                start_t -= cls.ecog_window_shift_sec
                step = cls.ecog_window_step_sec
                n = ecog_window_n
            else:
                #start_t += (cls.ecog_window_shift_sec *.5)

                # start_t += (cls.ecog_window_shift_sec * 2)
                # start_t = ixes.median() #- cls.ecog_window_shift_sec
                step = cls.ecog_window_step_sec #* 1.25
                n = ecog_window_n  # * 2

            sample_indices[wrd_id] = [label_index
                                          .loc[start_t + i * step:]
                                          .iloc[:win_size]
                                          .index
                                      for i in range(n)]

            # Hacky, but just remove anything that's not the right side from the end
            sample_indices[wrd_id] = [ixes for ixes in sample_indices[wrd_id]
                                      if len(ixes) == win_size]
            if len(sample_indices[wrd_id]) == 0:
                msg = ("Error: word %d had no windows" % wrd_id)
                #raise ValueError(msg)

        return sample_indices

    @staticmethod
    def stim_adjust__power_threshold(data_map, threshold=0.007):
        audio_s = data_map['audio']
        stim_s = data_map['stim']

        rolling_pwr = audio_s.abs().rolling(48000).max().reindex(stim_s.index).fillna(method='ffill')
        stim_auto_m = (stim_s != 0.) & (rolling_pwr > threshold)
        # Subtract one for no speech (0)
        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[stim_auto_m].nunique(False))
            print(msg)
        stim_pwrt_s = pd.Series(np.where(stim_auto_m, stim_s, 0), index=stim_s.index)
        stim_pwrt_diff_s = stim_pwrt_s.diff().fillna(0).astype(int)

        updates = dict(stim_pwrt=stim_pwrt_s, stim_pwrt_diff=stim_pwrt_diff_s,
                       rolling_audio_pwr=rolling_pwr)
        remaps = dict(stim='stim_pwrt', stim_diff='stim_pwrt_diff')
        return updates, remaps
        #data_map.update(updates)
        #data_map['remap']['stim'] = 'stim_pwrt'
        #data_map['remap']['stim_diff'] = 'stim_pwrt_diff'
        #return data_map

    @staticmethod
    def stim_adjust__power_quantile(data_map, q=0.75):
        audio_s = data_map['audio']
        stim_s = data_map['stim']

        rolling_pwr = audio_s.abs().rolling(48000).mean().reindex(stim_s.index).fillna(method='ffill').fillna(0)
        q_thresh_s = rolling_pwr.groupby(stim_s).quantile(q)
        q_thresh_s.loc[0] = 0
        stim_auto_m = (stim_s != 0.) & (rolling_pwr >= stim_s.map(q_thresh_s.to_dict()))

        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[stim_auto_m].nunique(False))
            print(msg)
        stim_pwrq_s = pd.Series(np.where(stim_auto_m, stim_s, 0), index=stim_s.index)
        stim_pwrq_diff_s = stim_pwrq_s.diff().fillna(0).astype(int)


        updates = dict(stim_pwrq=stim_pwrq_s, stim_pwrq_diff=stim_pwrq_diff_s,
                       rolling_audio_pwr=rolling_pwr)
        remaps = dict(stim='stim_pwrq', stim_diff='stim_pwrq_diff')
        return updates, remaps

    @staticmethod
    def _add_and_remap_data_element(data_map, k_v_tuple, replaces):
        assert k_v_tuple[0] not in data_map
        data_map[k_v_tuple[0]] = k_v_tuple[1]
        data_map['remap'][replaces] = k_v_tuple[0]
        return data_map

    #@staticmethod
    #def add__stim_diff(data_map):
    #    stim_s = data_map['remap']['stim']
    @staticmethod
    def default_preprocessing(data_map, verbose=False):
        stim_s = data_map[data_map['remap']['stim']]
        ######
        # Stim event codes and txt
        # 0 is neutral, so running difference will identify the onsets
        stim_diff_s = stim_s.diff().fillna(0).astype(int)
        #return dict(stim_diff=stim_diff_s)

        word_code_d = data_map[data_map['remap']['word_code_d']]
        ######
        # Stim event codes and txt
        # 0 is neutral, so running difference will identify the onsets
        #stim_diff_s = stim_s.diff().fillna(0).astype(int)

        # Grab only the samples at the onset
        start_times_cd = stim_diff_s.loc[stim_diff_s > 0].astype(int)
        # start_times_txt = start_times_cd.map(lambda v: word_code_arr[v - 1])
        start_times_txt = start_times_cd.map(word_code_d)

        stop_times_cd = stim_diff_s.loc[stim_diff_s < 0].astype(int).abs()
        # stop_times_txt = stop_times_cd.map(lambda v: word_code_arr[v - 1])
        stop_times_txt = stop_times_cd.map(word_code_d)

        ####
        start_time_map = {f"{i}-{w}": st
                          for i, (st, w) in enumerate(start_times_txt.to_dict().items())}
        stop_time_map = {f"{i}-{w}": st
                         for i, (st, w) in enumerate(stop_times_txt.to_dict().items())}

        if verbose:
            print(f"Sample Words: {','.join(list(start_time_map.keys())[:5])}")

        updates = dict(stim_diff=stim_diff_s,
         #stim_auto=stim_auto_s,
         #stim_auto_diff=stim_auto_diff_s,
         start_times_d=start_time_map,
         stop_times_d=stop_time_map)
        #data_map.update(updates)
        return updates, dict()

        #data_map['remap']
        #return dict(start_time_map=start_time_map, stop_time_map=stop_time_map)

    @classmethod
    def process_ecog(cls, nww, ix):
        return nww.ecog_df.loc[ix.min():].dropna().iloc[:nww.max_ecog_window_size]

    @classmethod
    def make_sample_object(cls, ix, text_label, ecog_df,
                           speech_df, #max_ecog_samples,
                           ecog_transform=None,
                           fields=('ecog_arr', 'text_arr'),
                           mfcc_f=None):

        kws = dict()
        # TODO: Reindexing may cause problems here - need tolerance param?
        # kws['ecog'] = ecog_df.reindex(ix).dropna()
        if 'ecog_arr' in fields:
            #kws['ecog'] = ecog_df.loc[ix.min() - cls.ecog_window_shift_sec:].iloc[:max_ecog_samples]
            kws['ecog'] = ecog_df.loc[ix ]
            # Transpose to keep time as last index for torch
            np_ecog_arr = kws['ecog'].values.T
            if ecog_transform is not None:
                #print("Apply transform to shape of " + str(np_ecog_arr.shape))
                np_ecog_arr = ecog_transform(np_ecog_arr)
            kws['ecog_arr'] = torch.from_numpy(np_ecog_arr).float()  # / 10.
        # Can't do this for non-contihous label regions
        # ecog=nww.ecog_df.loc[ixes.min(): ixes.max()],
        # Word index is base on wave file, so just reindex

        if 'speech_arr' in fields:
            kws['speech'] = speech_df.reindex(ix)
            kws['speech_arr'] = torch.from_numpy(kws['speech'].values).float()

        if 'mfcc_arr' in fields:
            mfcc_f = (torchaudio.transforms.MFCC(cls.audio_sample_rate)
                      if mfcc_f is None else mfcc_f)
            kws['mfcc_arr'] = mfcc_f(kws['speech_arr'])

        if 'text_arr' in fields:
            kws['text'] = '<silence>' if text_label <= 0 else '<speech>'
            # kws['test'] = text_enc_f()
            kws['text_arr'] = torch.Tensor({'<silence>': [0], '<speech>': [1]}.get(kws['text']))

        kws.update({k: None for k in cls.schema_fields if k not in kws})
        s = cls.sample_schema(**kws)
        return s

    #######
    ## Path handling
    @classmethod
    def set_default_base_path(cls, p):
        cls.default_base_path = p

    @staticmethod
    def make_filename(patient, session, trial, location='Mayo Clinic'):
        if location == 'Mayo Clinic':
            return f"MC{str(patient).zfill(3)}-SW-S{session}-R{trial}.mat"
        else:
            raise ValueError("Don't know location " + location)

    @classmethod
    def get_data_path(cls, patient, session, trial, location='Mayo Clinic', base_path=None):
        fname = cls.make_filename(patient, session, trial, location)
        base_path = cls.default_base_path if base_path is None else base_path
        p = os.path.join(base_path, location, 'Data', fname)
        return p

    ######
    # Conversion of MATLAB data structure to map of pandas data objects
    @staticmethod
    def parse_mat_arr_dict(mat_d, sensor_columns=None, verbose=True):
        fs_audio = mat_d['fs_audio'][0][0]
        fs_signal = mat_d['fs_signal'][0][0]
        stim_arr = mat_d['stimcode'].reshape(-1).astype('int32')

        # Create a dictonary map from index to word string repr
        # **0 is neutral, word index starts from 1**?
        word_code_d = {i + 1: w[0] for i, w in enumerate(mat_d['wordcode'].reshape(-1))}
        # Code 0 as no-sound/signal/speech
        word_code_d[0] = '<ns>'

        ########
        # Check for repeated words by parsing to Series
        word_code_s = pd.Series(word_code_d, name='word')
        word_code_s.index.name = 'word_index'

        w_vc = word_code_s.value_counts()
        dup_words = w_vc[w_vc > 1].index.tolist()

        if verbose:
            print("Duplicate words (n=%d): %s"
                  % (len(dup_words), ", ".join(dup_words)))

        for dw in dup_words:
            for w_ix in word_code_s[word_code_s == dw].index.tolist():
                new_wrd = dw + ("-%d" % w_ix)
                word_code_d[w_ix] = new_wrd
                # if verbose: print(new_wrd)

        word_code_s = pd.Series(word_code_d, name='word')
        word_code_s.index.name = 'word_index'

        ######
        # Stim parse
        ix = pd.TimedeltaIndex(pd.RangeIndex(0, stim_arr.shape[0]) / fs_signal, unit='s')
        stim_s = pd.Series(stim_arr, index=ix)
        ecog_df = pd.DataFrame(mat_d['ECOG_signal'], index=ix)
        if verbose:
            print(f"ECOG shape: {ecog_df.shape} [{ecog_df.index[0], ecog_df.index[-1]}]")

        #####
        # Channels/sensors status
        # TODO: What are appropriate names for these indicators
        chann_code_cols = ["code_%d" % e for e in range(mat_d['electrodes'].shape[-1])]
        channel_df = pd.DataFrame(mat_d['electrodes'], columns=chann_code_cols)

        # TODO: Use channel columns to set ?
        sensor_columns = channel_df.index.tolist() if sensor_columns is None else sensor_columns
        ch_m = (channel_df['code_0'] == 1)
        sensor_columns = [c for c in ch_m[ch_m].index.tolist() if c in sensor_columns]
        print("Selected sensors: " + (", ".join(map(str, sensor_columns))))

        ######
        # Audio
        audio_arr = mat_d['audio'].reshape(-1)
        ix = pd.TimedeltaIndex(pd.RangeIndex(0, audio_arr.shape[0]) / fs_audio, unit='s')
        audio_s = pd.Series(audio_arr, index=ix)
        if verbose:
            print(f"Audio shape: {audio_s.shape} [{audio_s.index[0], audio_s.index[-1]}]")

        ####
        # TESTING AUTO ADJUSTING MASK
        #rolling_pwr = audio_s.abs().rolling(48000).max()
        #stim_auto_m = (stim_s != 0.) & (rolling_pwr.reindex(stim_s.index).fillna(method='ffill') > 0.005)
        #eq = stim_s.nunique(dropna=False) == stim_s[stim_auto_m].nunique(dropna=False)
        #if not eq:
        #    msg = "stim_s and stim_auto not equal: %d != %d" % (stim_s.nunique(False),
        #                                                        stim_s[stim_auto_m].nunique(False))
        #    print(msg)
        #stim_auto_s = pd.Series(np.where(stim_auto_m, stim_s, 0), index=stim_s.index)
        #stim_auto_diff_s = stim_auto_s.diff().fillna(0).astype(int)



        ret_d = dict(ecog_all=ecog_df,
                     ecog=ecog_df.loc[:, sensor_columns],
                     audio=audio_s,
                     channel_status=channel_df,
                     stim=stim_s,
                     sensor_columns=sensor_columns,
                     #stim_diff=stim_diff_s,
                     #stim_auto=stim_auto_s,
                     #stim_auto_diff=stim_auto_diff_s,
                     #start_times_d=start_time_map,
                     #stop_times_d=stop_time_map,
                     word_code_d=word_code_d,
                     )
        ret_d['remap'] = {k:k for k in ret_d.keys()}
        return ret_d

    #######
    # Entry point to get data
    @classmethod
    def load_data(cls, location=None, patient=None, session=None, trial=None, base_path=None,
                  parse_mat_data=True, sensor_columns=None, verbose=True):

        location = 'Mayo Clinic' if location is None else location
        patient = 19 if patient is None else patient
        location = 1 if location is None else location
        session = 1 if session is None else session
        sensor_columns = cls.sensor_columns if sensor_columns is None else sensor_columns

        if verbose:
            print(f"---{patient}-{session}-{trial}-{location}---")

        p = cls.get_data_path(patient, session, trial, location, base_path)
        import scipy.io
        mat_d = scipy.io.matlab.loadmat(p)
        if parse_mat_data:
            return cls.parse_mat_arr_dict(mat_d, sensor_columns=sensor_columns,
                                          verbose=verbose)

        return mat_d


@attr.s
class NorthwesternWords_BCI2k(BaseDataset):
    env_key = 'NORTHWESTERNWORDS_DATASET'
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 pkg_data_dir))
    base_path = attr.ib(None)
    # num_mfcc = 13
    sensor_columns = list(range(64))
    audio_sample_rate = 48000
    ecog_sample_rate = 1200
    ecog_pass_band = attr.ib((70, 250))

    # Hack
    ecog_window_shift_sec = pd.Timedelta(0.15, 's')
    ecog_window_step_sec = pd.Timedelta(0.01, 's')
    ecog_window_n = 40

    # In terms of audio samples
    fixed_window_size = attr.ib(audio_sample_rate * 1)
    fixed_window_step = attr.ib(int(audio_sample_rate * .01))
    max_ecog_window_size = attr.ib(600)
    num_mfcc = attr.ib(13)

    # TODO: make the fields mappable from a dictionary of functions
    # Trying this out: define a small data schema class
    schema_fields = ["ecog", "ecog_arr",
                     "speech", "speech_arr",
                     # "mfcc",
                     "mfcc_arr",
                     "text", "text_arr"]
    sample_schema = attr.make_class("NorthwesterWordsSample",
                                    schema_fields)
    sample_schema2 = attr.make_class("NorthwesterWordsSample2",
                                     {f: attr.ib(None) for f in schema_fields})
    selected_word_indices = attr.ib(None)

    def __attrs_post_init__(self):
        self.ecog_df = self.load_dat_to_frame(base_path=self.base_path)
        self.speech_df = self.load_wav_to_frame(base_path=self.base_path)

        self.s_ecog_df = self.ecog_df[self.sensor_columns]
        self.s_ecog_df = self.s_ecog_df.pipe(feature_processing.filter,
                                             band=self.ecog_pass_band,
                                             sfreq=self.ecog_sample_rate)

        # Important piece - identifies time segments automatically
        #   - Positive values are word/speech labels
        #   - Negative values are silence before the speech (i.e. -1 before 1
        self.word_index = feature_processing.compute_speech_index(self.speech_df)

        self.mfcc_m = torchaudio.transforms.MFCC(self.audio_sample_rate,
                                                 self.num_mfcc)

        #
        if self.selected_word_indices is not None:
            self.word_index = self.word_index[self.word_index.isin(self.selected_word_indices)]
        self.sample_indices_map = self.make_sample_indices(self.word_index,
                                                           win_size=self.fixed_window_size,
                                                           win_step=self.fixed_window_step)
        self.i_to_ix_key = {i: k for i, k in enumerate(self.sample_indices_map.keys())}
        self.sample_map = dict()

    def __len__(self):
        return len(self.sample_indices_map)

    def __getitem__(self, item):
        ix_k = self.i_to_ix_key[item]

        # if ix_k not in self.sample_map:
        #    self.sample_map[ix_k] = self.make_sample_object(self.sample_indices_map[ix_k],
        #                                                    ix_k[0], self.ecog_df[self.sensor_columns],
        #                                                    self.speech_df, max_ecog_samples=self.max_ecog_window_size,
        #                                                    mfcc_f=self.mfcc_m)
        # so = self.sample_map[ix_k]
        so = self.make_sample_object(self.sample_indices_map[ix_k],
                                     ix_k[0], self.s_ecog_df,
                                     self.speech_df, max_ecog_samples=self.max_ecog_window_size,
                                     mfcc_f=self.mfcc_m)

        return {k: v for k, v in attr.asdict(so).items()
                if isinstance(v, torch.Tensor)}

    def sample_plot(self, i, band=None,
                    offset_seconds=0,
                    figsize=(15, 10), axs=None):
        import matplotlib
        from IPython.display import display
        # offs = pd.Timedelta(offset_seconds)
        t_word_ix = self.word_index[self.word_index == i].index
        offs_td = pd.Timedelta(offset_seconds, 's')
        t_word_slice = slice(t_word_ix.min() - offs_td, t_word_ix.max() - offs_td)
        display(t_word_slice)
        display(t_word_ix.min() - offs_td)
        # t_word_ix = self.word_index.loc[t_word_ix.min() - offs_td: t_word_ix.max() - offs_td].index
        # t_word_ecog_df = self.ecog_df.reindex(t_word_ix).dropna()
        # t_word_wav_df = self.speech_df.reindex(t_word_ix)
        t_word_ecog_df = self.ecog_df.loc[t_word_slice].dropna()
        t_word_wav_df = self.speech_df.loc[t_word_slice]
        display(t_word_ecog_df.describe())
        scols = self.sensor_columns

        ecog_std = self.ecog_df[scols].std()
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=ecog_std.min(), vmax=ecog_std.max())
        c_f = lambda v: cmap(norm(v))
        colors = ecog_std.map(c_f).values

        if axs is None:
            fig, axs = matplotlib.pyplot.subplots(figsize=figsize, nrows=2)
        else:
            fig = axs[0].get_figure()

        if band is not None:
            plt_df = t_word_ecog_df[scols].pipe(feature_processing.filter, band=band,
                                                sfreq=self.ecog_sample_rate)
        else:
            plt_df = t_word_ecog_df[scols]

        ax = plt_df.plot(alpha=0.3, legend=False,
                         color=colors, lw=1.2,
                         ax=axs[0], fontsize=14)
        ax.set_title(f"{len(plt_df)} samples")

        ax = t_word_wav_df.plot(alpha=0.7, legend=False, fontsize=14, ax=axs[1])
        ax.set_title(f"{len(t_word_wav_df)} samples, word index = {i}")

        fig.tight_layout()

        return axs

    @classmethod
    def make_sample_indices(cls, label_index, win_size, win_step):
        sample_indices = dict()
        grp = label_index.groupby(label_index)
        for wrd_id, wave_wrd_values in grp:
            ixes = wave_wrd_values.index
            # ixes = label_index.index
            # if win_size > len(ixes):
            start_t = ixes.min()
            if wrd_id > 0:
                start_t -= cls.ecog_window_shift_sec
                step = cls.ecog_window_step_sec
                n = cls.ecog_window_n
            else:
                start_t += (cls.ecog_window_shift_sec * 1)
                # start_t += (cls.ecog_window_shift_sec * 2)
                # start_t = ixes.median() #- cls.ecog_window_shift_sec
                step = cls.ecog_window_step_sec * 1.25
                n = cls.ecog_window_n  # * 2

            for i in range(n):
                sl = slice(start_t + i * step, None)
                sample_indices[(wrd_id, i)] = label_index.loc[sl].iloc[:win_size].index
            # elif len(ix_iter) == 0:
            #    pass
            # else:
            #    ix_iter = range(0, len(ixes) - win_size, win_step)
            #    if len(ix_iter) == 0:
            #        print(f'{wrd_id} has not ixes')

            #    for _ix in ix_iter:
            #        _ixes = ixes[_ix:_ix + win_size]
            #        sample_indices[(wrd_id, _ix)] = _ixes
            #        if wrd_id > 0:
            #            print(f'{wrd_id} has some entries')
        return sample_indices

    @classmethod
    def process_ecog(cls, nww, ix):
        return nww.ecog_df.loc[ix.min():].dropna().iloc[:nww.max_ecog_window_size]

    @classmethod
    def make_sample_object(cls, ix, text_label, ecog_df,
                           speech_df, max_ecog_samples,
                           fields=('ecog_arr', 'text_arr'),
                           mfcc_f=None):

        kws = dict()
        # TODO: Reindexing may cause problems here - need tolerance param?
        # kws['ecog'] = ecog_df.reindex(ix).dropna()
        if 'ecog_arr' in fields:
            kws['ecog'] = ecog_df.loc[ix.min() - cls.ecog_window_shift_sec:].iloc[:max_ecog_samples]
            # Transpose to keep time as last index for torch
            kws['ecog_arr'] = torch.from_numpy(kws['ecog'].values.T).float()  # / 10.
        # Can't do this for non-contihous label regions
        # ecog=nww.ecog_df.loc[ixes.min(): ixes.max()],
        # Word index is base on wave file, so just reindex

        if 'speech_arr' in fields:
            kws['speech'] = speech_df.reindex(ix)
            kws['speech_arr'] = torch.from_numpy(kws['speech'].values).float()

        if 'mfcc_arr' in fields:
            mfcc_f = (torchaudio.transforms.MFCC(cls.audio_sample_rate)
                      if mfcc_f is None else mfcc_f)
            kws['mfcc_arr'] = mfcc_f(kws['speech_arr'])

        if 'text_arr' in fields:
            kws['text'] = '<silence>' if text_label <= 0 else '<speech>'
            # kws['test'] = text_enc_f()
            kws['text_arr'] = torch.Tensor({'<silence>': [0], '<speech>': [1]}.get(kws['text']))

        kws.update({k: None for k in cls.schema_fields if k not in kws})
        s = cls.sample_schema(**kws)
        return s

    #######
    @classmethod
    def set_default_base_path(cls, p):
        cls.default_base_path = p

    @staticmethod
    def make_filename(patient, session, recording):
        if isinstance(patient, int):
            patient = str(patient).zfill(3)

        if isinstance(session, int):
            session = str(session).zfill(3)

        if isinstance(recording, int):
            recording = str(recording).zfill(2)

        return "Pt{patient}S{session}R{recording}".format(patient=patient,
                                                          session=session,
                                                          recording=recording)

    @classmethod
    def get_path(cls, extension, patient=22, session=1, recording=1, base_path=None):
        fname = cls.make_filename(patient, session, recording) + extension
        base_path = cls.default_base_path if base_path is None else base_path
        p = os.path.join(base_path, fname)
        return p

    @classmethod
    def get_data_path(cls, patient=22, session=1, recording=1, base_path=None):
        return cls.get_path(extension=".dat", patient=patient, session=session,
                            recording=recording, base_path=base_path)

    @classmethod
    def get_wave_path(cls, patient=22, session=1, recording=1, base_path=None):
        return cls.get_path(extension=".wav", patient=patient, session=session,
                            recording=recording, base_path=base_path)

    @classmethod
    def load_dat_file(cls, patient=22, session=1, recording=1, base_path=None, as_frame=True):
        p = cls.get_data_path(patient=patient, session=session,
                              recording=recording, base_path=base_path)
        r = b2k.BCI2kReader(p)

        if as_frame:
            df = pd.DataFrame(r.signals.T)
            df.index = pd.TimedeltaIndex(df.index / cls.ecog_sample_rate, unit='s')

            for k, v in r.states.items():
                df[k] = v[0]
            r = df

        return r

    @classmethod
    def load_dat_to_frame(cls, patient=22, session=1, recording=1, base_path=None):
        return cls.load_dat_file(patient, session, recording, base_path, as_frame=True)

    @classmethod
    def load_wav_file(cls, patient=22, session=1, recording=1, base_path=None, as_frame=True):
        fname = cls.make_filename(patient, session, recording) + ".wav"
        base_path = cls.default_base_path if base_path is None else base_path
        p = os.path.join(base_path, fname)
        from scipy.io import wavfile
        r = wavfile.read(p)
        if as_frame:
            srate, wavdata = r
            assert srate == cls.audio_sample_rate
            wav_s = pd.Series(wavdata, name='wavdata')
            wav_s.index = pd.TimedeltaIndex(wav_s.index / srate, unit='s')
            wav_s.index.name = 'ts_%d_hz' % srate
            r = wav_s

        return r

    @classmethod
    def load_wav_to_frame(cls, patient=22, session=1, recording=1, base_path=None):
        return cls.load_wav_file(patient, session, recording, base_path, as_frame=True)

import matplotlib
import pandas as pd
from glob import glob
from os import path
import numpy as np
import torch
from torch.utils import data as tdata
from tqdm.auto import tqdm
import h5py
import scipy.io

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
    env_key = None
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

    @staticmethod
    def get_dataset_by_name(dataset_name):
        # Could be done with simple dictionary, but in function
        # can do minor processing on the name if needed
        if dataset_name == 'nww':
            dataset_cls = NorthwesternWords
        elif dataset_name == 'chang-nww':
            dataset_cls = ChangNWW
        else:
            raise ValueError("Unknown dataset: %s" % dataset_name)
        return dataset_cls


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
class ShuffleDimension:
    shuffle_dim = attr.ib(1)

    def __call__(self, sample):
        sample = np.copy(sample)
        # TODO/WARNING : probably won't work for more than 2d?
        # swap the shuffle_dim to the zeroth index
        sample = np.transpose(sample, [self.shuffle_dim, 0])
        # shuffle on the 0-dim in place
        np.random.shuffle(sample)
        # Swap the shuffle_dim back - i.e. do same transpose again
        sample = np.transpose(sample, [self.shuffle_dim, 0])
        return sample

@attr.s
class RandomIntLike:
    low = attr.ib(0)
    high = attr.ib(2)

    def __call__(self, sample):
        return torch.randint(self.low, self.high, sample.shape, device=sample.device).type_as(sample)

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
    data_subset = 'Data'# attr.ib('Data')
    mat_d_signal_key = 'ECOG_signal'
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

    mc_patient_set_map = {
        19: [('Mayo Clinic', 19, 1, 1),
             ('Mayo Clinic', 19, 1, 2),
             ('Mayo Clinic', 19, 1, 3)],

        21: [('Mayo Clinic', 21, 1, 1),
             ('Mayo Clinic', 21, 1, 2)],

        22: [('Mayo Clinic', 22, 1, 1),
             ('Mayo Clinic', 22, 1, 2),
             ('Mayo Clinic', 22, 1, 3)],

        24: [('Mayo Clinic', 24, 1, 2),
             ('Mayo Clinic', 24, 1, 3),
             ('Mayo Clinic', 24, 1, 4)],

        25: [('Mayo Clinic', 25, 1, 1),
             ('Mayo Clinic', 25, 1, 2)],

        26: [('Mayo Clinic', 26, 1, 1),
             ('Mayo Clinic', 26, 1, 2)],
    }
    all_patient_maps = dict(MC=mc_patient_set_map)

    # num_mfcc = 13
    #signal_key = 'signal'
    sensor_columns = attr.ib(None)
    default_sensor_columns = list(range(64))
    default_audio_sample_rate = 48000
    default_ecog_sample_rate = 1200
    ecog_pass_band = attr.ib((70, 250))

    # In terms of audio samples
    # fixed_window_size = attr.ib(audio_sample_rate * 1)
    # fixed_window_step = attr.ib(int(audio_sample_rate * .01))
    #ecog_window_size = attr.ib(300)
    num_mfcc = attr.ib(13)
    verbose = attr.ib(True)

    selected_word_indices = attr.ib(None)

    #stim_indexing_source = attr.ib('stim_diff')
    transform = attr.ib(None)
    target_transform = attr.ib(None)

    power_threshold = attr.ib(0.007)
    power_q = attr.ib(0.70)
    pre_processing_pipeline = attr.ib(None)
    # If using one source of data, with different `selected_word_indices`, then
    # passing the first NWW dataset to all subsequent ones built on the same source data
    # can save on memory and reading+parsing time
    data_from: 'NorthwesternWords' = attr.ib(None)

    ###
    # Add new processing pipelines for NWW here
    def make_pipeline_map(self, default='minimal'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """
        #samp_ix = feature_processing.SampleIndicesFromStim(ecog_window_size=self.ecog_window_size,
        #                                                  ecog_window_n=self.ecog_window_n,
        #                                                  ecog_window_step_sec=self.ecog_window_step_sec,
        #                                                  ecog_window_shift_sec=self.ecog_window_shift_sec
        #                                                   )
        self.sample_ixer = feature_processing.ChangSampleIndicesFromStim()

        p_map = {
            'threshold':
                (
                    feature_processing.SubsampleECOG() >>
                    feature_processing.WordStopStartTimeMap() >>
                    feature_processing.PowerThreshold(threshold=self.power_threshold) >>
                    self.sample_ixer),
            'quantile':
                (
                        feature_processing.SubsampleECOG() >>
                        feature_processing.WordStopStartTimeMap() >>
                 feature_processing.PowerQuantile(q=self.power_q) >>
                 self.sample_ixer),
            'minimal':

                feature_processing.SubsampleECOG() >>
                feature_processing.WordStopStartTimeMap() >> self.sample_ixer
        }
        p_map['default'] = p_map[default]
        return p_map

    def __attrs_post_init__(self):
        # Build pipelines based on this NWW dataset state
        self.pipeline_map = self.make_pipeline_map()

        # If nothing passed, use 'default' pipeline
        if self.pre_processing_pipeline is None:
            self.pipeline_f = self.pipeline_map['default']
        # If string passed, use it to select the pipeline in the map
        elif isinstance(self.pre_processing_pipeline, str):
            self.pipeline_f = self.pipeline_map[self.pre_processing_pipeline]
        # Otherwise, just assume it will work, that a list of tuple(callable, kws) was passed
        else:
            self.pipeline_f = self.pre_processing_pipeline

        # If no data sharing, then load and parse data from scratch
        if self.data_from is None:
            # Leave this here for now...
            #self.mfcc_m = torchaudio.transforms.MFCC(self.default_audio_sample_rate,
            #                                         self.num_mfcc)

            ## Data loading ##
            # - Load the data, parsing into pandas data frame/series types
            # - Only minimal processing into Python objects done here
            data_iter = tqdm(self.patient_tuples, desc="Loading data")
            mat_data_maps = {l_p_s_t_tuple: self.load_data(*l_p_s_t_tuple,
                                                            sensor_columns=self.sensor_columns,
                                                            parse_mat_data=False,
                                                            verbose=self.verbose)
                              for l_p_s_t_tuple in data_iter}
            good_and_bad_tuple_d = {l_p_s_t_tuple: self.identify_good_and_bad_sensors(mat_d, self.sensor_columns)
                                        for l_p_s_t_tuple, mat_d in mat_data_maps.items()}
            self.selected_columns = sorted(list({_gs for k, (gs, bs) in good_and_bad_tuple_d.items()
                                                 for _gs in gs}))
            self.sensor_count = len(self.selected_columns)
            #data_iter = tqdm(self.patient_tuples, desc="parsing data")
            self.data_maps = {l_p_s_t_tuple: self.parse_mat_arr_dict(mat_d, self.selected_columns)
                              for l_p_s_t_tuple, mat_d in tqdm(mat_data_maps.items(), desc='Parsing data')}
            ###-----

            ## Sensor check ##
            # Get selected sensors from each dataset into map
            #self.sensor_map = {k: dmap['sensor_columns']
            #                   for k, dmap in self.data_maps.items()}
            # Make unique set and assert that they are all the same length
            #self.sensor_counts = list(set(map(len, self.sensor_map.values())))
            #if len(self.sensor_counts) == 1:
            #    # Once validated that all sensor columns same length, set it as attribute
            #    self.sensor_count = self.sensor_counts[0]
            #    self.selected_columns = list(self.sensor_map.values())[0]
            #else:
            #    #raise NotImplementedError("underlying datasets have different number of sensor columns")
            #    print("Warning: sensor columns don't match - will try to use superset")
            #    unique_sensors = sorted(list({c for k, cols in self.sensor_map.items() for c in cols}))
            #    self.sensor_count = len(unique_sensors)
            #    self.selected_columns = unique_sensors

            assert self.sensor_count == len(self.selected_columns)
            print(f"Selected {len(self.selected_columns)} sensors")
                #for
            ###-----

            ## Important processing ##
            # - Process each subject in data map through pipeline func
            self.sample_index_maps = dict()
            for k in self.data_maps.keys():
                dmap = self.data_maps[k]
                res_dmap = self.pipeline_f(dmap)
                self.data_maps[k] = res_dmap
                self.sample_index_maps[k] = res_dmap['sample_index_map']
                self.fs_signal = getattr(self, 'fs_signal', res_dmap['fs_signal'])
                self.ecog_window_size = getattr(self, 'ecog_window_size',
                                                int(self.fs_signal * self.sample_ixer.window_size.total_seconds()))
                if self.fs_signal != res_dmap['fs_signal']:
                    raise ValueError("Mismatch fs (%s!=%s) on %s" % (self.fs_signal, res_dmap['fs_signal'], str(k)))

            # Map full description (word label, window index,...trial key elements..}
            # to the actual pandas index
            self.flat_index_map = {tuple([wrd_id, ix_i] + list(k_t)): ixes
                                   for k_t, index_map in self.sample_index_maps.items()
                                   for wrd_id, ix_list in index_map.items()
                                   for ix_i, ixes in enumerate(ix_list)}

            # Enumerate all the keys across flat_index_map into one large list for index-style,
            # has a len() and can be indexed into nicely (via np.ndarray)
            self.flat_keys = np.array([(k, k[2:])
                                       for i, k in enumerate(self.flat_index_map.keys())],
                                      dtype='object')

        else:
            print("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            #self.mfcc_m = self.data_from.mfcc_m
            self.data_maps = self.data_from.data_maps
            self.sample_index_maps = self.data_from.sample_index_maps
            self.flat_index_map = self.data_from.flat_index_map
            self.flat_keys = self.data_from.flat_keys

        self.select(self.selected_word_indices)

    def __len__(self):
        return len(self.selected_flat_keys)

    def __getitem__(self, item):
        # ix_k includes the class and window id
        # data_k specifies subject dataset in data_map (less granular than ix_k)
        ix_k, data_k = self.selected_flat_keys[item]
        data_d = self.data_maps[data_k]

        # Put it all together (TODO: cleanup this interface)
        # TODO: Make features, make target methods?
        #so = self.make_sample_object(self.flat_index_map[ix_k],
        #                             ix_k[0], data_d['ecog'],
        #                             data_d['audio'],
        #                             ecog_transform=self.transform,
        #                             #max_ecog_samples=self.max_ecog_window_size,
        #                             mfcc_f=self.mfcc_m)
        so = self.get_features(data_d, self.flat_index_map[ix_k],
                               ix_k[0], ecog_transform=self.transform)
        so.update(self.get_targets(data_d, self.flat_index_map[ix_k],
                                   ix_k[0], target_transform=self.target_transform))

        # Return anything that is a Torch Tensor - the torch dataloader will handle
        # compiling multiple outputs for batch
        return {k: v for k, v in so.items()
                if isinstance(v, torch.Tensor)}

    def select(self, sample_indices):
        # select out specific samples from the flat_keys array if selection passed
        # - Useful if doing one-subject training and want to split data up among datasets for use
        self.selected_word_indices = sample_indices
        if self.selected_word_indices is not None:
            self.selected_flat_keys = self.flat_keys[self.selected_word_indices]
        else:
            self.selected_flat_keys = self.flat_keys

        return self

    @staticmethod
    def get_features(data_map, ix, label, ecog_transform=None):
        ecog_df = data_map['ecog']
        kws = dict()

        kws['ecog'] = ecog_df.loc[ix]
        # Transpose to keep time as last index for torch
        np_ecog_arr = kws['ecog'].values.T
        if ecog_transform is not None:
            # print("Apply transform to shape of " + str(np_ecog_arr.shape))
            np_ecog_arr = ecog_transform(np_ecog_arr)
        kws['ecog_arr'] = torch.from_numpy(np_ecog_arr).float()  # / 10.
        return kws

    @classmethod
    def identify_good_and_bad_sensors(cls, mat_d, sensor_columns=None, ):
        if 'electrodes' in mat_d:
            chann_code_cols = ["code_%d" % e for e in range(mat_d['electrodes'].shape[-1])]
            channel_df = pd.DataFrame(mat_d['electrodes'], columns=chann_code_cols)
            print("Found electrodes metadata, N trodes = %d" % channel_df.shape[0] )

            #required_sensor_columns = channel_df.index.tolist() if sensor_columns is None else sensor_columns
            # Mask for good sensors
            ch_m = (channel_df['code_0'] == 1)
            all_valid_sensors = ch_m[ch_m].index.tolist()

            # Spec the number of sensors that the ecog array mush have
            if sensor_columns is None:
                required_sensor_columns = channel_df.index.tolist()
            elif sensor_columns == 'valid':
                sensor_columns = all_valid_sensors
                required_sensor_columns = sensor_columns
            else:
                required_sensor_columns = sensor_columns

                #
            good_sensor_columns = [c for c in all_valid_sensors if c in required_sensor_columns]
            bad_sensor_columns = list(set(required_sensor_columns) - set(good_sensor_columns))

                #if len(bad_sensor_columns) == 0:
                #    print("No bad sensors")
                #elif bad_sensor_method == 'zero' and len(bad_sensor_columns) > 0:
                #    print("Zeroing %d bad sensor columns: %s" % (len(bad_sensor_columns), str(bad_sensor_columns)))
                #    ecog_df.loc[:, bad_sensor_columns] = 0.
                #elif bad_sensor_method == 'ignore':
                #    print("Ignoring bad sensors")
                #else:
                #    raise ValueError("Unknown bad_sensor_method (use 'zero', 'ignore'): " + str(bad_sensor_method))
        else:
            good_sensor_columns = None
            bad_sensor_columns = None

        return good_sensor_columns, bad_sensor_columns
            #channel_df = None
            #sensor_columns = ecog_df.columns.tolist() if sensor_columns is None else sensor_columns
            #print(f"No 'electrods' key in mat data - using all {len(sensor_columns)} columns")
            #ch_m = ecog_df.columns.notnull()


    @staticmethod
    def get_targets(data_map, ix, label, target_transform=None):
        kws = dict(text='<silence>' if label <= 0 else '<speech>',
                   text_arr=torch.Tensor([0] if label <= 0 else [1]))
        if target_transform is not None:
            kws['text_arr'] = target_transform(kws['text_arr'])
        return kws

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
        t_word_slice = slice(t_word_ix.min() - offs_td, t_word_ix.max() + offs_td)
        display(t_word_slice)
        display(t_word_ix.min() - offs_td)
        # t_word_ix = self.word_index.loc[t_word_ix.min() - offs_td: t_word_ix.max() - offs_td].index
        # t_word_ecog_df = self.ecog_df.reindex(t_word_ix).dropna()
        # t_word_wav_df = self.speech_df.reindex(t_word_ix)
        ecog_df = self.data_maps[data_k]['ecog']
        speech_df = self.data_maps[data_k]['audio']
        word_txt = self.data_maps[data_k]['word_code_d'].get(ix_k[0], '<no speech>')

        t_word_ecog_df = ecog_df.loc[t_word_slice].dropna()
        t_word_wav_df = speech_df.loc[t_word_slice]
        #display(t_word_ecog_df.describe())
        scols = self.default_sensor_columns

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
                                                sfreq=self.fs_signal)
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

        if 'text_arr' in fields:
            kws['text'] = '<silence>' if text_label <= 0 else '<speech>'
            kws['text_arr'] = torch.Tensor([0] if text_label <= 0 else [1])
            # kws['test'] = text_enc_f()
            #kws['text_arr'] = torch.Tensor({'<silence>': [0], '<speech>': [1]}.get(kws['text']))

        return kws

    #######
    ## Path handling
    @staticmethod
    def make_filename(patient, session, trial, location='Mayo Clinic'):
        if location == 'Mayo Clinic':
            return f"MC{str(patient).zfill(3)}-SW-S{session}-R{trial}.mat"
        else:
            raise ValueError("Don't know location " + location)

    @classmethod
    def get_data_path(cls, patient, session, trial, location='Mayo Clinic',
                      subset=None, base_path=None):
        fname = cls.make_filename(patient, session, trial, location)
        base_path = cls.default_base_path if base_path is None else base_path
        subset = cls.data_subset if subset is None else subset
        p = os.path.join(base_path, location, subset, fname)
        return p

    ######
    # Conversion of MATLAB data structure to map of pandas data objects
    @classmethod
    def parse_mat_arr_dict(cls, mat_d, sensor_columns=None,
                           zero_repr='<ns>', defaults=None,
                           bad_sensor_method='zero',
                           verbose=True):
        """
        Convert a raw matlab dataset into Python+Pandas with timeseries indices

        Parameters

        mat_d: dict()
            Dictionary of data returned from scip.io matlab load
        sensor_columns : list()
            List of sensor IDs to use
        zero_repr : string
            String code for no-speech/class
        defaults : dict()
            Values are generally taken from the matlab dataset,
            followed by this defaults dict, followed by the class
            static default values
        verbose : boolean
            Print extra information


        Returns : dict
            Extracted and wrangled data and configurations
        """
        if defaults is None:
            defaults = dict()

        try:
            fs_audio = mat_d['fs_audio'][0][0]
        except KeyError:
            #fs_audio = cls.audio_sample_rate
            fs_audio = defaults.get('fs_audio',
                                    cls.default_audio_sample_rate)

        #assert fs_audio == cls.default_audio_sample_rate
        print("Audio FS = " + str(fs_audio))

        try:
            fs_signal = mat_d['fs_signal'][0][0]
        except KeyError:
            fs_signal = defaults.get('fs_signal',
                                     cls.default_ecog_sample_rate)

        stim_arr = mat_d['stimcode'].reshape(-1).astype('int32')

        # Create a dictonary map from index to word string repr
        # **0 is neutral, word index starts from 1**?
        word_code_d = {i + 1: w[0] for i, w in enumerate(mat_d['wordcode'].reshape(-1))}
        # Code 0 as no-sound/signal/speech
        word_code_d[0] = zero_repr

        ########
        # Check for repeated words by parsing to Series
        word_code_s = pd.Series(word_code_d, name='word')
        word_code_s.index.name = 'word_index'

        w_vc = word_code_s.value_counts()
        dup_words = w_vc[w_vc > 1].index.tolist()

        if verbose:
            print("Duplicate words (n=%d): %s"
                  % (len(dup_words), ", ".join(dup_words)))

        # Re-write duplicate words with index so they never collide
        for dw in dup_words:
            for w_ix in word_code_s[word_code_s == dw].index.tolist():
                new_wrd = dw + ("-%d" % w_ix)
                word_code_d[w_ix] = new_wrd
                # if verbose: print(new_wrd)

        # Recreate the word code series to include the new words
        word_code_s = pd.Series(word_code_d, name='word')
        word_code_s.index.name = 'word_index'

        ######
        # Stim parse
        ix = pd.TimedeltaIndex(pd.RangeIndex(0, stim_arr.shape[0]) / fs_signal, unit='s')
        stim_s = pd.Series(stim_arr, index=ix)
        ecog_df = pd.DataFrame(mat_d[cls.mat_d_signal_key], index=ix)
        if verbose:
            print(f"{cls.mat_d_signal_key} shape: {ecog_df.shape} [{ecog_df.index[0], ecog_df.index[-1]}]")

        ######
        # Stim event codes and txt
        # 0 is neutral, so running difference will identify the onsets
        stim_diff_s = stim_s.diff().fillna(0).astype(int)

        #####
        # Channels/sensors status
        # TODO: What are appropriate names for these indicators
        if 'electrodes' in mat_d:
            chann_code_cols = ["code_%d" % e for e in range(mat_d['electrodes'].shape[-1])]
            channel_df = pd.DataFrame(mat_d['electrodes'], columns=chann_code_cols)
            print("Found electrodes metadata, N trodes = %d" % channel_df.shape[0] )

            #required_sensor_columns = channel_df.index.tolist() if sensor_columns is None else sensor_columns
            # Mask for good sensors
            ch_m = (channel_df['code_0'] == 1)
            all_valid_sensors = ch_m[ch_m].index.tolist()

            # Spec the number of sensors that the ecog array mush have
            if sensor_columns is None:
                required_sensor_columns = channel_df.index.tolist()
            elif sensor_columns == 'valid':
                sensor_columns = all_valid_sensors
                required_sensor_columns = sensor_columns
            else:
                required_sensor_columns = sensor_columns

                #
                good_sensor_columns = [c for c in all_valid_sensors if c in required_sensor_columns]
                bad_sensor_columns = list(set(required_sensor_columns) - set(good_sensor_columns))
                if len(bad_sensor_columns) == 0:
                    print("No bad sensors")
                elif bad_sensor_method == 'zero' and len(bad_sensor_columns) > 0:
                    print("Zeroing %d bad sensor columns: %s" % (len(bad_sensor_columns), str(bad_sensor_columns)))
                    ecog_df.loc[:, bad_sensor_columns] = 0.
                elif bad_sensor_method == 'ignore':
                    print("Ignoring bad sensors")
                else:
                    raise ValueError("Unknown bad_sensor_method (use 'zero', 'ignore'): " + str(bad_sensor_method))


        else:
            channel_df = None
            sensor_columns = ecog_df.columns.tolist() if sensor_columns is None else sensor_columns
            print(f"No 'electrods' key in mat data - using all {len(sensor_columns)} columns")
            #ch_m = ecog_df.columns.notnull()

        print(f"Selected sensors (n={len(sensor_columns)}): "
              + (", ".join(map(str, sensor_columns))))

        ######
        # Audio
        if 'audio' in mat_d:
            audio_arr = mat_d['audio'].reshape(-1)
            ix = pd.TimedeltaIndex(pd.RangeIndex(0, audio_arr.shape[0]) / fs_audio, unit='s')
            audio_s = pd.Series(audio_arr, index=ix)
            if verbose:
                print(f"Audio shape: {audio_s.shape} [{audio_s.index[0], audio_s.index[-1]}]")
        else:
            #audio = None
            audio_s = None

        ####
        # TESTING AUTO ADJUSTING MASK
        ret_d = dict(
            fs_audio=fs_audio, fs_signal=fs_signal,
            ecog_all=ecog_df,
                     ecog=ecog_df.loc[:, sensor_columns],
                     audio=audio_s,
                     channel_status=channel_df,
                     stim=stim_s,
                     stim_diff=stim_diff_s,
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
                  parse_mat_data=True, sensor_columns=None, bad_sensor_method='zero',
                  verbose=True):

        location = 'Mayo Clinic' if location is None else location
        patient = 19 if patient is None else patient
        location = 1 if location is None else location
        session = 1 if session is None else session
        trial = 1 if trial is None else trial
        sensor_columns = cls.default_sensor_columns if sensor_columns is None else sensor_columns

        if verbose:
            print(f"---{patient}-{session}-{trial}-{location}---")

        p = cls.get_data_path(patient, session, trial, location, base_path=base_path)
        mat_d = scipy.io.matlab.loadmat(p)
        if parse_mat_data:
            return cls.parse_mat_arr_dict(mat_d, sensor_columns=sensor_columns,
                                          bad_sensor_method=bad_sensor_method,
                                          verbose=verbose)

        return mat_d


    @classmethod
    def plot_word_sample_region(cls, data_map, word_code=None, figsize=(15, 5), plot_features=False,
                                subplot_kwargs=None,
                                feature_key='ecog', feature_ax=None, ax=None):
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
                     .resample('5ms').first().fillna(method='ffill'))

        silence_s = pd.Series(0, index=plt_audio.index)
        silence_s.loc[silence_min_ix : silence_max_ix] = 0.95

        speaking_s = pd.Series(0, index=plt_audio.index)
        speaking_s.loc[speaking_min_ix : speaking_max_ix] = 0.95

        #####
        feature_ax = None
        splt_kws = dict() if subplot_kwargs is None else subplot_kwargs
        if not plot_features and ax is None:
            fig, ax = matplotlib.pyplot.subplots(figsize=figsize, **splt_kws)
        elif not plot_features:
            fig = ax.get_figure()
        elif plot_features and ax is None or feature_ax is None:
            fig, (ax, feature_ax) = matplotlib.pyplot.subplots(figsize=figsize, nrows=2, **splt_kws)

        ax = plt_audio.plot(legend=False, alpha=0.4, color='tab:grey', figsize=(15, 5), label='audio', ax=ax)
        ax.set_title(f"Min-ts={plt_min} || Max-ts={plt_max}\n\
        Labeled Regions: word_code={word_code}, word='{data_map['word_code_d'][word_code]}'\
        \nSpeaking N windows={len(t_speaking_ixes)}; Silence N windows={len(t_speaking_ixes)}")
        ax2 = ax.twinx()

        ax2.set_ylim(0.05, 1.1)
        # ax.axvline(silence_min_ix / pd.Timedelta(1,'s'))
        #(data_map['stim'].reindex(data_map['audio'].index).fillna(method='ffill').loc[plt_min: plt_max] > 0).astype(
        #    int).plot(ax=ax2, color='tab:blue', label='original stim')
        (data_map['stim'].resample('5ms').first().fillna(method='ffill').loc[plt_min: plt_max] > 0).astype(
            int).plot(ax=ax2, color='tab:blue', label='original stim')

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

    @classmethod
    def make_tuples_from_sets_str(cls, sets_str):
        """
        Process a string representation of the patient tuples, e.g.: 'MC-19-0,MC-19-1'
        """
        if sets_str is None:
            return None

        # e.g. MC-19-0
        if ',' in sets_str:
            sets_str_l = sets_str.split(',')
            # Recurse
            return [cls.make_tuples_from_sets_str(s)[0] for s in sets_str_l]

        org, pid, ix = sets_str.split('-')
        assert pid.isdigit() and ix.isdigit() and org in ('MC',)
        pmap, pid, ix = cls.all_patient_maps[org], int(pid), int(ix)
        assert pid in pmap
        p_list = pmap[pid]
        return [p_list[ix]]

    # TODO: This may be better placed with the Trainer class, as that's where most
    #       of the model+dataset coupling happens. In other words, the dataset shouldn't
    #       have to know how a model is going to use it, but the trainer must know this to train
    def eval_model(self, model, win_step=1, device=None):
        model_preds = dict()
        print(f"Running {len(self.data_maps)} eval data map(s): {', '.join(map(str, self.data_maps.keys()))}")
        for ptuple, data_map in self.data_maps.items():
            ecog_torch_arr = torch.from_numpy(data_map['ecog'].values).float()
            # TODO: seems like there should be a better way to do this
            all_ecog_dl = torch.utils.data.DataLoader([ecog_torch_arr[_ix:_ix + self.ecog_window_size].T
                                                       for _ix in
                                                       range(0, ecog_torch_arr.shape[0] - self.ecog_window_size, win_step)],
                                                      batch_size=1024, num_workers=6)
            with torch.no_grad():
                # TODO: cleaner way to handle device
                if device is None:
                    all_ecog_out = [model(x) for x in tqdm(all_ecog_dl)]
                else:
                    all_ecog_out = [model(x.to(device)) for x in tqdm(all_ecog_dl)]

            # TODO: When we take a windows prediction - seems like it should represent
            # the prediction at the end of that window...?
            ix_range = range(self.ecog_window_size, ecog_torch_arr.shape[0], win_step)
            #ix_range = range(0, ecog_torch_arr.shape[0] - self.ecog_window_size, win_step)
            all_ecog_pred_s = pd.Series([_v.item() for v in all_ecog_out for _v in v],
                                        index=data_map['ecog'].iloc[ix_range].index,
                                        name='pred_proba')
            model_preds[ptuple] = all_ecog_pred_s

        return model_preds

@attr.s
class ChangNWW(NorthwesternWords):
    #data_subset = 'Preprocessed/Chang1'
    data_subset = 'Preprocessed/Chang3'
    mat_d_signal_key = 'signal'
    default_ecog_sample_rate = 200
    patient_tuples = attr.ib(
        (('Mayo Clinic', 19, 1, 2),)
    )
    #ecog_window_size = attr.ib(100)
    #ecog_window_shift_sec = pd.Timedelta(0.75, 's')
    #ecog_window_step_sec = attr.ib(0.01, factory=lambda s: pd.Timedelta(s, 's'))
    #ecog_window_step_sec = pd.Timedelta(0.01, 's')
    #ecog_window_n = attr.ib(60)

    # ecog samples
    ecog_window_size = attr.ib(100)

    pre_processing_pipeline = attr.ib('minimal')
    data_from: 'NorthwesternWords' = attr.ib(None)

    def make_pipeline_map(self, default='quantile'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """
        #samp_ix = feature_processing.SampleIndicesFromStim(ecog_window_size=self.ecog_window_size,
        #                                                   ecog_window_n=self.ecog_window_n,
        #                                                   ecog_window_step_sec=self.ecog_window_step_sec,
        #                                                   ecog_window_shift_sec=self.ecog_window_shift_sec)
        self.sample_ixer = feature_processing.ChangSampleIndicesFromStim()
        self.mfcc = feature_processing.MFCC(self.num_mfcc)

        p_map = {
            'threshold':
                (feature_processing.WordStopStartTimeMap() >>
                 feature_processing.PowerThreshold(threshold=self.power_threshold) >>
                 self.sample_ixer >> self.mfcc),
            'quantile':
                (feature_processing.WordStopStartTimeMap() >>
                 feature_processing.PowerQuantile(q=self.power_q) >>
                 self.sample_ixer >> self.mfcc),
            'minimal': feature_processing.WordStopStartTimeMap() >> self.sample_ixer >> self.mfcc
        }
        p_map['default'] = p_map[default]
        return p_map

    @staticmethod
    def get_features(data_map, ix, label, ecog_transform=None):
        kws = NorthwesternWords.get_features(data_map, ix, label, ecog_transform)
        return kws

    @staticmethod
    def get_targets(data_map, ix, label):
        kws = NorthwesternWords.get_targets(data_map, ix, label)
        kws['mfcc'] = torch.from_numpy(data_map['mfcc'].loc[ix].values).float()
        return kws

    @classmethod
    def load_data(cls, location=None, patient=None, session=None, trial=None, base_path=None,
                  parse_mat_data=True, sensor_columns=None, verbose=True):
        #kws = dict(location=lcoa
        #           )
        kws = locals()
        print(kws.keys())
        no_parse_kws = {k: False if k == 'parse_mat_data' else v
                        for k, v in kws.items()
                        if k not in ('cls', '__class__')}
        chang_data = super(ChangNWW, cls).load_data(**no_parse_kws)
        nww_data = NorthwesternWords.load_data(**no_parse_kws)

        chang_data['audio'] = nww_data['audio']

        if parse_mat_data:
            return  cls.parse_mat_arr_dict(chang_data,
                                           sensor_columns=cls.default_sensor_columns if sensor_columns is None else sensor_columns,
                                           verbose=verbose)
        return chang_data

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
        return nww.ecog_df.loc[ix.min():].dropna().iloc[:nww.ecog_window_size]

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
        from BCI2kReader import BCI2kReader as b2k
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

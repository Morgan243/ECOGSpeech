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
import logging

from os import environ
import os
import attr
#import torchaudio
import socket
from ecog_speech import feature_processing, utils, pipeline
from sklearn.pipeline import  Pipeline

with_logger = utils.with_logger(prefix_name=__name__)

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')

os.environ['WANDB_CONSOLE'] = 'off'

#########
# Torchvision style transformations for use in the datasets and loaders
@attr.s
class RollDimension:
    """
    Shift all values in a dimension by a random amount, values "rolled off"
    reappaer at the opposite side.
    """
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
    """
    Shuffle a dimension of the input sample
    """
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
    """
    Produce random integers in the specified range with the same shape
    as the input sample
    """
    low = attr.ib(0)
    high = attr.ib(2)

    def __call__(self, sample):
        return torch.randint(self.low, self.high, sample.shape, device=sample.device).type_as(sample)


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


class StanfordTasks(BaseDataset):
    """Place holder for standford data"""
    pass


@attr.s
class DEAP(BaseDataset):
    """WIP - Incomplete"""
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
@with_logger
class BaseASPEN(BaseDataset):
    env_key = None
    default_base_path = None
    all_patient_maps = None
    default_sensor_columns = list(range(64))
    default_audio_sample_rate = 48000
    default_signal_sample_rate = 1200

    mat_d_keys = dict(
        signal=None,
        signal_fs=None,
        audio='audio',
        audio_fs='fs_audio',
        stimcode='stimcode',
        electrodes='electrodes',
        wordcode='wordcode')

    patient_tuples = attr.ib(None)
    sensor_columns = attr.ib(None)
    base_path = attr.ib(None)
    data_subset = attr.ib('Data')

    num_mfcc = attr.ib(13)
    verbose = attr.ib(True)

    selected_word_indices = attr.ib(None)
    transform = attr.ib(None)
    target_transform = attr.ib(None)

    power_threshold = attr.ib(0.007)
    power_q = attr.ib(0.70)
    pre_processing_pipeline = attr.ib(None)
    # If using one source of data, with different `selected_word_indices`, then
    # passing the first NWW dataset to all subsequent ones built on the same source data
    # can save on memory and reading+parsing time
    data_from: 'NorthwesternWords' = attr.ib(None)

    default_data_subset = 'Data'
    default_location = None
    default_patient = None
    default_session = None
    default_trial = None

    def __attrs_post_init__(self):
        # Build pipelines based on this NWW dataset state
        self.pipeline_map = self.make_pipeline_map()

        # If nothing passed, use 'default' pipeline
        if self.pre_processing_pipeline is None:
            self.logger.info("Default pipeline selected")
            self.pipeline_f = self.pipeline_map['default']
        # If string passed, use it to select the pipeline in the map
        elif isinstance(self.pre_processing_pipeline, str):
            self.logger.info(f"{self.pre_processing_pipeline} pipeline selected")
            self.pipeline_f = self.pipeline_map[self.pre_processing_pipeline]
        # Otherwise, just assume it will work, that a callable is passed
        # TODO: Check that pipeline_f is callable
        else:
            self.logger.info(f"{str(self.pre_processing_pipeline)} pipeline passed directly")
            self.pipeline_f = self.pre_processing_pipeline

        # If no data sharing, then load and parse data from scratch
        if self.data_from is None:
            self.logger.info("Loading data directly")
            # Leave this here for now...
            #self.mfcc_m = torchaudio.transforms.MFCC(self.default_audio_sample_rate,
            #                                         self.num_mfcc)

            ## Data loading ##
            # - Load the data, parsing into pandas data frame/series types
            # - Only minimal processing into Python objects done here
            data_iter = tqdm(self.patient_tuples, desc="Loading data")
            mat_data_maps = {l_p_s_t_tuple: self.load_data(*l_p_s_t_tuple,
                                                            #sensor_columns=self.sensor_columns,
                                                           # IMPORTANT: Don't parse data yet
                                                            parse_mat_data=False,
                                                            subset=self.data_subset,
                                                            verbose=self.verbose)
                              for l_p_s_t_tuple in data_iter}

            ## Important processing ##
            # - Process each subject in data map through pipeline func
            self.sample_index_maps = dict()
            self.data_maps = dict()
            #for k in self.data_maps.keys():
            for k, dmap in mat_data_maps.items():
                res_dmap = self.pipeline_f(dmap)
                self.sample_index_maps[k] = res_dmap['sample_index_map']
                self.fs_signal = getattr(self, 'fs_signal', res_dmap[self.mat_d_keys['signal_fs']])
                #self.ecog_window_size = getattr(self, 'ecog_window_size',
                #                                int(self.fs_signal * self.sample_ixer.window_size.total_seconds()))
                #self.ecog_window_size = int(self.fs_signal * self.sample_ixer.window_size.total_seconds())
                self.n_samples_per_window = res_dmap['n_samples_per_window']
                self.logger.info(f"N samples per window: {self.n_samples_per_window}")

                if self.fs_signal != res_dmap[self.mat_d_keys['signal_fs']]:
                    raise ValueError("Mismatch fs (%s!=%s) on %s" % (self.fs_signal, res_dmap['fs_signal'], str(k)))

                self.data_maps[k] = res_dmap

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

            ###
            # Sensor selection logic - based on the patients loaded - which sensors do we use?
            if self.sensor_columns is None or isinstance(self.sensor_columns, str):
                #good_and_bad_tuple_d = {l_p_s_t_tuple: self.identify_good_and_bad_sensors(mat_d, self.sensor_columns)
                #                            for l_p_s_t_tuple, mat_d in mat_data_maps.items()}
                good_and_bad_tuple_d = {l_p_s_t_tuple: (mat_d['good_sensor_columns'], mat_d['bad_sensor_columns'])
                                        for l_p_s_t_tuple, mat_d in self.data_maps.items()}

                good_and_bad_tuple_d = {k: (set(gs) if gs else (list(range(self.data_maps[k][self.mat_d_keys['signal']].shape[1]))),
                                            bs)
                                         for k, (gs, bs) in good_and_bad_tuple_d.items()}
                #print("GOOD AND BAD SENSORS: " + str(good_and_bad_tuple_d))
                self.logger.info("GOOD AND BAD SENSORS: " + str(good_and_bad_tuple_d))
                self.sensor_columns = 'union' if self.sensor_columns is None else self.sensor_columns
                # UNION: Select all good sensors from all inputs, zeros will be filled for those missing
                if self.sensor_columns == 'union':
                    self.selected_columns = sorted(list({_gs for k, (gs, bs) in good_and_bad_tuple_d.items()
                                                        for _gs in gs}))
                # INTERSECTION: Select only sensors that are rated good in all inputs
                elif self.sensor_columns == 'intersection' or self.sensor_columns == 'valid':
                    s = [set(gs) for k, (gs, bs) in good_and_bad_tuple_d.items()]
                    self.selected_columns = sorted(list(s[0].intersection(*s[1:])))

                #elif self.sensor_columns == 'all':
                else:
                    raise ValueError("Unknown snsor columns argument: " + str(self.sensor_columns))
                #print("Selected columns with -%s- method: %s"
                #      % (self.sensor_columns, ", ".join(map(str, self.selected_columns))) )
                self.logger.info("Selected columns with -%s- method: %s"
                                  % (self.sensor_columns, ", ".join(map(str, self.selected_columns))) )
            else:
                self.selected_columns = self.sensor_columns
            self.sensor_count = len(self.selected_columns)

            # Finish processing the data mapping loaded from the mat data files
            #self.data_maps = {l_p_s_t_tuple: self.parse_mat_arr_dict(mat_d, self.selected_columns)
            #                  for l_p_s_t_tuple, mat_d in tqdm(mat_data_maps.items(), desc='Parsing data')}
            ###-----
            assert self.sensor_count == len(self.selected_columns)
            #print(f"Selected {len(self.selected_columns)} sensors")
            self.logger.info(f"Selected {len(self.selected_columns)} sensors")
            ###-----

            self.sensor_selection_trf = skl_feature_processing.ApplySensorSelection(selection=self.selected_columns)
            self.data_maps = {l_p_s_t_tuple: self.sensor_selection_trf.transform(mat_d)
                              for l_p_s_t_tuple, mat_d in tqdm(self.data_maps.items(),
                                                               desc='Applying sensor selection')}


        else:
            #print("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            self.logger.warning("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            #self.mfcc_m = self.data_from.mfcc_m
            self.data_maps = self.data_from.data_maps
            self.sample_index_maps = self.data_from.sample_index_maps
            self.flat_index_map = self.data_from.flat_index_map
            self.flat_keys = self.data_from.flat_keys

        self.select(self.selected_word_indices)

    def make_pipeline_map(self, default='audio_gate'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """

        p_map = {
            'audio_gate': Pipeline([
                ('parse_signal', skl_feature_processing.ParseTimeSeriesArrToFrame(self.mat_d_keys['signal'],
                                                                                 self.mat_d_keys['signal_fs'],
                                                                                 1200, output_key='signal')),
                ('parse_audio', skl_feature_processing.ParseTimeSeriesArrToFrame(self.mat_d_keys['audio'],
                                                                                  self.mat_d_keys['audio_fs'],
                                                                                  48000, reshape=-1)),
                ('parse_stim', skl_feature_processing.ParseTimeSeriesArrToFrame(self.mat_d_keys['stimcode'],
                                                                                 self.mat_d_keys['signal_fs'],
                                                                                 1200, reshape=-1, output_key='stim')),
                ('sensor_selection', skl_feature_processing.IdentifyGoodAndBadSensors(sensor_selection=self.sensor_columns)),
                ('subsample', skl_feature_processing.SubsampleSignal()),
                ('Threshold', skl_feature_processing.PowerThreshold(speaking_window_samples=48000 // 16,
                                                                    silence_window_samples=int(48000 * 1.5),
                                                                    speaking_quantile_threshold=0.9,
                                                                    #silence_threshold=0.001,
                                                                    #silGence_quantile_threshold=0.05,
                                                                    silence_n_smallest=5000)),
                ('speaking_indices', skl_feature_processing.WindowSampleIndicesFromStim('stim_pwrt',
                                                                                    target_onset_shift=pd.Timedelta(-.5, 's'),
                                                                                    # input are centers, and output is a window of .5 sec
                                                                                    # so to center it, move the point (center) back .25 secods
                                                                                    # so that extracted 0.5 sec window saddles the original center
                                                                                    #target_offset_shift=pd.Timedelta(-0.25, 's')
                                                                                    target_offset_shift=pd.Timedelta(-0.5, 's')
                                                                                    )
                 ),

                ('silence_indices', skl_feature_processing.WindowSampleIndicesFromIndex('silence_stim_pwrt_s',
                                                                                    # Center the extracted 0.5 second window
                                                                                    index_shift=pd.Timedelta(-0.25, 's'),
                                                                                    stim_value_remap=0
                                                                                  )),
                ('output', 'passthrough')
                    ]).transform,

            'minimal':
                feature_processing.SubsampleECOG() >>
                feature_processing.WordStopStartTimeMap() >> feature_processing.ChangSampleIndicesFromStim()
        }
        p_map['default'] = p_map[default]
        return p_map

    def to_eval_replay_dataloader(self, patient_k=None, win_step=1, batch_size=1024, num_workers=4,
                                  ecog_transform=None):
        if patient_k is None:
            patient_k = list(self.data_maps.keys())
        elif not isinstance(patient_k, list):
            patient_k = [patient_k]

        dl_map = dict()
        for k in patient_k:
            data_map = self.data_maps[k]
            ecog_torch_arr = torch.from_numpy(data_map['ecog'].values).float()
            outputs = list()
            for _iix in tqdm(range(0, ecog_torch_arr.shape[0] - self.ecog_window_size, win_step),
                            desc='creating windows'):
                _ix = slice(_iix, _iix + self.ecog_window_size)
                feats = self.get_features(data_map, _ix, ecog_transform=ecog_transform, index_loc=True)
                # TODO: Just grabbing the max stim wode in the range - better or more useful way to do this?
                targets = self.get_targets(data_map, None, label=data_map['stim'].iloc[_ix].max())
                so = dict(**feats, **targets)
                so = {k: v for k, v in so.items()
                        if isinstance(v, torch.Tensor)}
                outputs.append(so)
            t_dl = torch.utils.data.DataLoader(outputs, batch_size=batch_size, num_workers=num_workers)
            dl_map[k] = t_dl

        ret = dl_map
        #if len(ret) == 1:
        #    ret = list(dl_map.values())[0]
        return ret

    def __len__(self):
        return len(self.selected_flat_keys)

    def __getitem__(self, item):
        # ix_k includes the class and window id
        # data_k specifies subject dataset in data_map (less granular than ix_k)
        ix_k, data_k = self.selected_flat_keys[item]
        data_d = self.data_maps[data_k]

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

    ######
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
        try:
            mat_dat_map = scipy.io.loadmat(p)
        except NotImplementedError as e:
            #cls.logger.info(f"Couldn't load {p} with scipy (vers > 7.3?) - manually loading as H5PY")
            cls.logger.info(f"Couldn't load {p} with scipy (vers > 7.3?) - using package 'mat73' to load")
            import mat73
            mat_dat_map = mat73.loadmat(p)
            #with h5py.File(p, 'r') as f:
            #    mat_dat_map = {k: np.array(f[k]) for k in f.keys()}
        return mat_dat_map

    # Conversion of MATLAB data structure to map of pandas data objects
#    @classmethod
#    def REMOVE_parse_mat_arr_dict(cls, mat_d, sensor_columns=None,
#                           zero_repr='<ns>', defaults=None,
#                           bad_sensor_method='zero',
#                           verbose=True) -> dict:
#        """
#        Convert a raw matlab dataset into Python+Pandas with timeseries indices
#
#        Parameters
#
#        mat_d: dict()
#            Dictionary of data returned from scip.io matlab load
#        sensor_columns : list()
#            List of sensor IDs to use
#        zero_repr : string
#            String code for no-speech/class
#        defaults : dict()
#            Values are generally taken from the matlab dataset,
#            followed by this defaults dict, followed by the class
#            static default values
#        verbose : boolean
#            Print extra information
#
#
#        Returns : dict
#            Extracted and wrangled data and configurations
#        """
#        if defaults is None:
#            defaults = dict()
#
#        #try:
#        #    fs_audio = mat_d[cls.mat_d_keys['audio_fs']][0][0]
#        #except IndexError:
#        #    fs_audio = int(mat_d[cls.mat_d_keys['audio_fs']])
#        #except KeyError:
#        #    #fs_audio = cls.audio_sample_rate
#        #    fs_audio = defaults.get(cls.mat_d_keys['audio_fs'],
#        #                            cls.default_audio_sample_rate)
#
#        ##assert fs_audio == cls.default_audio_sample_rate
#        #cls.logger.info("Audio FS = " + str(fs_audio))
#
#        #try:
#        #    fs_signal = mat_d[cls.mat_d_keys['signal_fs']][0][0]
#        #except IndexError:
#        #    fs_signal = int(mat_d[cls.mat_d_keys['signal_fs']])
#        #except KeyError:
#        #    fs_signal = defaults.get(cls.mat_d_keys['signal_fs'],
#        #                             cls.default_signal_sample_rate)
#
#        #stim_arr = mat_d[cls.mat_d_keys['stimcode']].reshape(-1).astype('int32')
#
#        ## Create a dictonary map from index to word string repr
#        ## **0 is neutral, word index starts from 1**?
#        #if cls.mat_d_keys['wordcode'] is not None:
#        #    word_code_d = {i + 1: w[0] for i, w in enumerate(mat_d[cls.mat_d_keys['wordcode']].reshape(-1))}
#        ## Code 0 as no-sound/signal/speech
#        #    word_code_d[0] = zero_repr
#        #else:
#        #    word_code_d = None
#
#        ########
#        # Check for repeated words by parsing to Series
#        #word_code_s = pd.Series(word_code_d, name='word')
#        #word_code_s.index.name = 'word_index'
#
#        #w_vc = word_code_s.value_counts()
#        #dup_words = w_vc[w_vc > 1].index.tolist()
#
#        #if verbose:
#        #    cls.logger.info("Duplicate words (n=%d): %s"
#        #          % (len(dup_words), ", ".join(dup_words)))
#
#        ## Re-write duplicate words with index so they never collide
#        #for dw in dup_words:
#        #    for w_ix in word_code_s[word_code_s == dw].index.tolist():
#        #        new_wrd = dw + ("-%d" % w_ix)
#        #        word_code_d[w_ix] = new_wrd
#        #        # if verbose: print(new_wrd)
#
#        ## Recreate the word code series to include the new words
#        #word_code_s = pd.Series(word_code_d, name='word')
#        #word_code_s.index.name = 'word_index'
#
#        ######
#        # Stim parse
#        #ix = pd.TimedeltaIndex(pd.RangeIndex(0, stim_arr.shape[0]) / fs_signal, unit='s')
#        #stim_s = pd.Series(stim_arr, index=ix)
#        #signal_df = pd.DataFrame(mat_d[cls.mat_d_keys['signal']], index=ix)
#        #if verbose:
#        #cls.logger.debug(f"{cls.mat_d_keys['signal']} shape: {signal_df.shape} [{signal_df.index[0], signal_df.index[-1]}]")
#
#        ######
#        # Stim event codes and txt
#        # 0 is neutral, so running difference will identify the onsets
#        #stim_diff_s = stim_s.diff().fillna(0).astype(int)
#
#        #####
#        # Channels/sensors status
#        # TODO: What are appropriate names for these indicators
#        bad_sensor_columns = mat_d.get('bad_sensor_columns', list())
#        #electrodes = mat_d.get(cls.mat_d_keys['electrodes'])
#        #if electrodes is not None:
#        #    chann_code_cols = ["code_%d" % e for e in range(electrodes.shape[-1])]
#        #    channel_df = pd.DataFrame(electrodes, columns=chann_code_cols)
#        #    cls.logger.info("Found electrodes metadata, N trodes = %d" % channel_df.shape[0] )
#
#        #    #required_sensor_columns = channel_df.index.tolist() if sensor_columns is None else sensor_columns
#        #    # Mask for good sensors
#        #    ch_m = (channel_df['code_0'] == 1)
#        #    all_valid_sensors = ch_m[ch_m].index.tolist()
#
#        #    # Spec the number of sensors that the ecog array mush have
#        #    if sensor_columns is None:
#        #        required_sensor_columns = channel_df.index.tolist()
#        #    elif sensor_columns == 'valid':
#        #        sensor_columns = all_valid_sensors
#        #        required_sensor_columns = sensor_columns
#        #    else:
#        #        required_sensor_columns = sensor_columns
#        #        #
#        #        good_sensor_columns = [c for c in all_valid_sensors if c in required_sensor_columns]
#        #        bad_sensor_columns = list(set(required_sensor_columns) - set(good_sensor_columns))
#        if len(bad_sensor_columns) == 0:
#            cls.logger.info("No bad sensors")
#        elif bad_sensor_method == 'zero' and len(bad_sensor_columns) > 0:
#            cls.logger.info("Zeroing %d bad sensor columns: %s" % (len(bad_sensor_columns), str(bad_sensor_columns)))
#            mat_d['signal'].loc[:, bad_sensor_columns] = 0.
#        elif bad_sensor_method == 'ignore':
#            cls.logger.info("Ignoring bad sensors")
#        else:
#            raise ValueError("Unknown bad_sensor_method (use 'zero', 'ignore'): " + str(bad_sensor_method))
#
#        else:
#            channel_df = None
#            if sensor_columns is None:
#                sensor_columns = signal_df.columns.tolist()
#            else:
#                missing_sensors = [s for s in sensor_columns if s not in signal_df.columns.tolist()]
#                if len(missing_sensors) > 0:
#                    signal_df.loc[:, missing_sensors] = 0.
#            #sensor_columns = ecog_df.columns.tolist() if sensor_columns is None else sensor_columns
#            cls.logger.info(f"No 'electrods' key in mat data - using all {len(sensor_columns)} columns")
#            #ch_m = ecog_df.columns.notnull()
#
#        cls.logger.info(f"Selected sensors (n={len(sensor_columns)}): "
#              + (", ".join(map(str, sensor_columns))))
#
#        ######
#        # Audio
#        audio_arr = mat_d.get(cls.mat_d_keys['audio'])
#        audio_s = None
#        if audio_arr is not None:
#            audio_arr = audio_arr.reshape(-1)
#            ix = pd.TimedeltaIndex(pd.RangeIndex(0, audio_arr.shape[0]) / fs_audio, unit='s')
#            audio_s = pd.Series(audio_arr, index=ix)
#            cls.logger.debug(f"Audio shape: {audio_s.shape} [{audio_s.index[0], audio_s.index[-1]}]")
#
#        ####
#        # TESTING AUTO ADJUSTING MASK
#        ret_d = dict(
#            #fs_audio=fs_audio, fs_signal=fs_signal,
#            ecog_all=signal_df,
#                     ecog=signal_df.loc[:, sensor_columns],
#                     audio=audio_s,
#                     channel_status=channel_df,
#                     stim=stim_s,
#                     stim_diff=stim_diff_s,
#                     sensor_columns=sensor_columns,
#            bad_sensor_columns=bad_sensor_columns,
#                     #stim_diff=stim_diff_s,
#                     #stim_auto=stim_auto_s,
#                     #stim_auto_diff=stim_auto_diff_s,
#                     #start_times_d=start_time_map,
#                     #stop_times_d=stop_time_map,
#                     #word_code_d=word_code_d,
#                     )
#        ret_d['remap'] = {k:k for k in ret_d.keys()}
#        return ret_d

    @classmethod
    def make_filename(cls, patient, session, trial, location):
        raise NotImplementedError()

    @classmethod
    def get_data_path(cls, patient, session, trial, location,
                      subset=None, base_path=None):
        fname = cls.make_filename(patient, session, trial, location)
        base_path = cls.default_base_path if base_path is None else base_path
        subset = cls.default_data_subset if subset is None else subset
        p = os.path.join(base_path, location, subset, fname)
        return p

    #######
    # Entry point to get data
    @classmethod
    def load_data(cls, location=None, patient=None, session=None, trial=None, base_path=None,
                  parse_mat_data=True, sensor_columns=None, bad_sensor_method='zero',
                  subset=None, verbose=True):

        location = cls.default_location if location is None else location
        patient = cls.default_patient if patient is None else patient
        #location = 1 if location is None else location
        session = cls.default_session if session is None else session
        trial = cls.default_trial if trial is None else trial
        sensor_columns = cls.default_sensor_columns if sensor_columns is None else sensor_columns

        if verbose:
            cls.logger.info(f"---{patient}-{session}-{trial}-{location}---")
            if subset is not None:
                cls.logger.info("\t->Using Subset: " + str(subset))

        p = cls.get_data_path(patient, session, trial, location, base_path=base_path, subset=subset)
        mat_d = cls.load_mat_from_path(p)
        #if parse_mat_data:
        #    return cls.parse_mat_arr_dict(mat_d, sensor_columns=sensor_columns,
        #                                  bad_sensor_method=bad_sensor_method,
        #                                  verbose=verbose)

        return mat_d

    @classmethod
    def make_tuples_from_sets_str(cls, sets_str):
        """
        Process a string representation of the patient tuples, e.g.: 'MC-19-0,MC-19-1'
        """
        if sets_str is None:
            return None

        # e.g. MC-19-0,MC-19-1
        if ',' in sets_str:
            sets_str_l = sets_str.split(',')
            # Recurse - returns a list, so combine all lists into one with `sum` reduction
            return sum([cls.make_tuples_from_sets_str(s) for s in sets_str_l], list())

        set_terms = sets_str.split('-')
        # e.g. MC-22-1 has three terms ('MC', 22, 1) selecting a specific trial of a specific participant
        if len(set_terms) == 3:
            #org, pid, ix = sets_str.split('-')
            org, pid, ix = set_terms
            assert pid.isdigit() and ix.isdigit() and org in cls.all_patient_maps.keys()
            pmap, pid, ix = cls.all_patient_maps[org], int(pid), int(ix)
            assert pid in pmap, f"PID: {pid} not in {org}'s known data map"
            p_list = [pmap[pid][ix]]
        # e.g. MC-22 will return tuples for all of MC-22's data
        elif len(set_terms) == 2:
            org, pid = set_terms
            assert pid.isdigit() and org in cls.all_patient_maps.keys()
            pmap, pid = cls.all_patient_maps[org], int(pid)
            assert pid in pmap, f"PID: {pid} not in {org}'s known data map"
            p_list = pmap[pid]
        return p_list

    @staticmethod
    def get_features(data_map, ix, label=None, ecog_transform=None, index_loc=False, signal_key='signal'):
        signal_df = data_map[signal_key]
        kws = dict()

        kws['signal'] = signal_df.loc[ix] if not index_loc else signal_df.iloc[ix]
        # Transpose to keep time as last index for torch
        np_ecog_arr = kws['signal'].values.T
        if ecog_transform is not None:
            # print("Apply transform to shape of " + str(np_ecog_arr.shape))
            np_ecog_arr = ecog_transform(np_ecog_arr)
        kws['signal_arr'] = torch.from_numpy(np_ecog_arr).float()
        return kws

    @classmethod
    def REMOVE_identify_good_and_bad_sensors(cls, mat_d, sensor_columns=None, ):

        if 'electrodes' in mat_d:
            chann_code_cols = ["code_%d" % e for e in range(mat_d['electrodes'].shape[-1])]
            channel_df = pd.DataFrame(mat_d['electrodes'], columns=chann_code_cols)
            cls.logger.info("Found electrodes metadata, N trodes = %d" % channel_df.shape[0] )

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
        else:
            good_sensor_columns = None
            bad_sensor_columns = None

        return good_sensor_columns, bad_sensor_columns

    def REMOVE_get_indices(self, key='ecog', only_longest=False):
        ix_map = {ptuple: dmap[key].index
                  for ptuple, dmap in self.data_maps.items()}

        if only_longest:
            ptuple = longest = longest_k = None
            for ptuple, ix in ix_map.items():
                if longest is None or len(ix) > len(longest):
                    longest, longest_k = ix, ptuple
            return ptuple, longest
        else:
            return ix_map

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
        ecog_df = self.data_maps[data_k]['signal']
        speech_df = self.data_maps[data_k]['audio']
        word_txt = self.data_maps[data_k]['word_code_d'].get(ix_k[0], '<no speech>')

        t_word_ecog_df = ecog_df.loc[t_word_slice].dropna()
        t_word_wav_df = speech_df.loc[t_word_slice]
        #display(t_word_ecog_df.describe())
        #scols = self.default_sensor_columns
        scols = self.selected_columns

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
    def make_filename(cls, patient, session, trial, location):
        raise NotImplementedError()





@attr.s
@with_logger
class HarvardSentences(BaseASPEN):
    """
    """
    #logger = utils.get_logger('ecog.' + __name__)

    env_key = 'HARVARDSENTENCES_DATASET'
    default_hvs_path = path.join(pkg_data_dir, 'HarvardSentences')
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 default_hvs_path))
    mat_d_keys = dict(
        signal='sEEG_signal',
        signal_fs='fs_signal',
        audio='audio',
        audio_fs='fs_audio',
        stimcode='stimcode',
        electrodes=None,
        wordcode=None,
    )

    all_patient_maps = dict(UCSD={
         4: [('UCSD', 4, 1)],
         5: [('UCSD', 5, 1)],
        10: [('UCSD', 10, 1)],
        18: [('UCSD', 18, 1)],
        19: [('UCSD', 19, 1)],
        22: [('UCSD', 22, 1)],
        28: [('UCSD', 28, 1)],
    })


    @classmethod
    def make_filename(cls, patient, session, trial, location):
        """
        UCSD04_Task_1.mat  UCSD10_Task_1.mat  UCSD19_Task_1.mat  UCSD28_Task_1.mat
        UCSD05_Task_1.mat  UCSD18_Task_1.mat  UCSD22_Task_1.mat
        """
        cls.logger.info("Harvard sentences only uses location and patient identifiers")
        loc_map = cls.all_patient_maps.get(location)
        if loc_map is None:
            raise KeyError(f"Valid locations: {list(cls.all_patient_maps.keys())}")

        fname = f"{location}{patient:02d}_Task_1.mat"

        return fname


@attr.s
@with_logger
class NorthwesternWords(BaseASPEN):
    """
    Northwestern-style data: one spoken word per cue, aligned brain and audio data

    This class can load multiple trails as once - ensuring correct windowing, but allowing
    for torch data sampling and other support.
    """

    env_key = 'NORTHWESTERNWORDS_DATASET'
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 path.join(pkg_data_dir,
                                                           'SingleWord')
                                                 ))
    mat_d_keys = dict(
        signal='ECOG_signal',
        signal_fs='fs_signal',
        audio='audio',
        audio_fs='fs_audio',
        stimcode='stimcode',
        electrodes='electrodes',
        wordcode='wordcode',
    )

    mc_patient_set_map = {
        19: [('MayoClinic', 19, 1, 1),
             ('MayoClinic', 19, 1, 2),
             ('MayoClinic', 19, 1, 3)],

        21: [('MayoClinic', 21, 1, 1),
             ('MayoClinic', 21, 1, 2)],

        22: [('MayoClinic', 22, 1, 1),
             ('MayoClinic', 22, 1, 2),
             ('MayoClinic', 22, 1, 3)],

        24: [('MayoClinic', 24, 1, 2),
             ('MayoClinic', 24, 1, 3),
             ('MayoClinic', 24, 1, 4)],

        #25: [('MayoClinic', 25, 1, 1),
        #     ('MayoClinic', 25, 1, 2)],

        26: [('MayoClinic', 26, 1, 1),
             ('MayoClinic', 26, 1, 2)],
    }

    nw_patient_set_map = {
        1: [
            ('Northwestern', 1, 1, 1),
            ('Northwestern', 1, 1, 2),
            # ('Northwestern', 1, 1, 3),
        ],
        2: [
            ('Northwestern', 2, 1, 1),
            ('Northwestern', 2, 1, 2),
            ('Northwestern', 2, 1, 3),
            ('Northwestern', 2, 1, 4),
        ],
        3: [
            ('Northwestern', 3, 1, 1),
            ('Northwestern', 3, 1, 2),
        ],
        4: [
            ('Northwestern', 4, 1, 1),
            ('Northwestern', 4, 1, 2),
        ],
        5: [
            ('Northwestern', 5, 1, 2),
            ('Northwestern', 5, 1, 3),
            ('Northwestern', 5, 1, 4),
        ],
        6: [
            ('Northwestern', 6, 1, 7),
            ('Northwestern', 6, 1, 9),
        ],
    }
    syn_patient_set_map = {
        1: [('Synthetic', 1, 1, 1)],
        2: [('Synthetic', 2, 1, 1)],
        3: [('Synthetic', 3, 1, 1)],
        4: [('Synthetic', 4, 1, 1)],
        5: [('Synthetic', 5, 1, 1)],
        6: [('Synthetic', 6, 1, 1),
            ('Synthetic', 6, 1, 2)],
        7: [('Synthetic', 7, 1, 1),
            ('Synthetic', 7, 1, 2)],
        8: [('Synthetic', 8, 1, 1),
            ('Synthetic', 8, 1, 2)],
    }
    all_patient_maps = dict(MC=mc_patient_set_map,
                            SN=syn_patient_set_map,
                            NW=nw_patient_set_map)
    fname_prefix_map = {'MayoClinic': 'MC', 'Synthetic': 'SN', 'Northwestern': 'NW'}
    tuple_to_sets_str_map = {t:f"{l}-{p}-{i}"
                             for l, p_d in all_patient_maps.items()
                              for p, t_l in p_d.items()
                               for i, t in enumerate(t_l)}

    #######
    ## Path handling
    @classmethod
    def make_filename(cls, patient, session, trial, location):
        if location in cls.fname_prefix_map:#== 'Mayo Clinic':
            return f"{cls.fname_prefix_map.get(location)}{str(patient).zfill(3)}-SW-S{session}-R{trial}.mat"
        else:
            raise ValueError("Don't know location " + location)





@attr.s
class ChangNWW(NorthwesternWords):
    """
    Northwester-style with Chang pre-processing steps
    """
    #data_subset = 'Preprocessed/Chang1'
    data_subset = 'Preprocessed/Chang3'
    mat_d_signal_key = 'signal'
    default_signal_sample_rate = 200
    patient_tuples = attr.ib(
        (('Mayo Clinic', 19, 1, 2),)
    )

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
    def get_features(data_map, ix, label, ecog_transform=None, index_loc=False):
        kws = NorthwesternWords.get_features(data_map, ix, label, ecog_transform, index_loc)
        return kws

    @staticmethod
    def get_targets(data_map, ix, label, target_transform=None):
        kws = NorthwesternWords.get_targets(data_map, ix, label, target_transform=target_transform)
        kws['mfcc'] = torch.from_numpy(data_map['mfcc'].loc[ix].values).float()
        return kws

    @classmethod
    def load_data(cls, location=None, patient=None, session=None, trial=None, base_path=None,
                  parse_mat_data=True, sensor_columns=None, bad_sensor_method='zero', verbose=True):
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
            return cls.parse_mat_arr_dict(chang_data,
                                          sensor_columns=cls.default_sensor_columns if sensor_columns is None else sensor_columns,
                                          verbose=verbose)
        return chang_data

import os
import attr
import socket
from glob import glob
from os import path
from os import environ

import pandas as pd
import numpy as np
import scipy.io

import torch
import torchvision.transforms
from torch.utils import data as tdata

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable

from typing import List, Optional, Type, ClassVar

from ecog_speech import feature_processing, utils, pipeline
from sklearn.pipeline import Pipeline

with_logger = utils.with_logger(prefix_name=__name__)

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')

os.environ['WANDB_CONSOLE'] = 'off'

logger = utils.get_logger(__name__)


@dataclass
class MultiSensorOptions:
    flatten_sensors_to_samples: bool = False
    """Sensors will bre broken up into sensors - inputs beceome (1, N-timesteps) samples (before batching)"""
    random_sensors_to_samples: bool = False


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
        # return torch.roll(sample, roll_amount, self.roll_dim)
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


class SelectFromDim:
    """Expects numpy arrays - use in Compose - models.base.Select can be used in Torch """

    def __init__(self, dim=0, index='random', keep_dim=True):
        super(SelectFromDim, self).__init__()
        self.dim = dim
        self.index = index
        self.keep_dim = keep_dim

    def __call__(self, x):
        ix = self.index
        if isinstance(self.index, str) and self.index == 'random':
            ix = np.random.randint(0, x.shape[self.dim])

        x = np.take(x, indices=ix, axis=self.dim)

        if self.keep_dim:
            x = np.expand_dims(x, self.dim)

        return x


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


@attr.s
class BaseDataset(tdata.Dataset):
    env_key = None
    fs_signal = attr.ib(None, init=False)

    def to_dataloader(self, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, **kwargs):
        dset = self
        if random_sample:
            if batches_per_epoch is None:
                # batches_per_epoch = len(dset) // batch_size
                batches_per_epoch = int(np.ceil(len(dset) / batch_size))

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
        elif dataset_name == 'hvs':
            dataset_cls = HarvardSentences
        elif dataset_name == 'hvsmfc':
            dataset_cls = HarvardSentencesMFC
        else:
            raise ValueError("Unknown dataset: %s" % dataset_name)
        return dataset_cls

    def get_feature_shape(self):
        raise NotImplementedError()

    def get_target_shape(self):
        raise NotImplementedError()


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

    selected_word_indices = attr.ib(None)
    transform = attr.ib(None)
    transform_l = attr.ib(attr.Factory(list))
    target_transform = attr.ib(None)
    target_transform_l = attr.ib(attr.Factory(list))

    flatten_sensors_to_samples = attr.ib(False)
    extra_output_keys = attr.ib(None)
    post_processing_func = attr.ib(None)
    post_processing_kws = attr.ib(attr.Factory(dict))

    # power_threshold = attr.ib(0.007)
    # power_q = attr.ib(0.70)
    pre_processing_pipeline = attr.ib(None)
    # If using one source of data, with different `selected_word_indices`, then
    # passing the first NWW dataset to all subsequent ones built on the same source data
    # can save on memory and reading+parsing time
    data_from: 'BaseASPEN' = attr.ib(None)

    initialize_data = attr.ib(True)
    selected_flat_keys = attr.ib(None, init=False)

    default_data_subset = 'Data'
    default_location = None
    default_patient = None
    default_session = None
    default_trial = None

    def __attrs_post_init__(self):
        self.logger.debug(f"preparing pipeline")
        # Build pipelines based on this NWW dataset state
        self.pipeline_map = self.make_pipeline_map()
        self.logger.debug(f"Available pipelines: {list(self.pipeline_map.keys())}")

        if self.initialize_data:
            self.initialize()

    def initialize(self):

        # If nothing passed, use 'default' pipeline
        if self.pre_processing_pipeline is None:
            self.logger.info("Default pipeline selected")
            self.pipeline_f = self.pipeline_map['default']
        # If string passed, use it to select the pipeline in the map
        elif isinstance(self.pre_processing_pipeline, str):
            self.logger.info(f"'{self.pre_processing_pipeline}' pipeline selected")
            self.pipeline_f = self.pipeline_map[self.pre_processing_pipeline]
        # Otherwise, just assume it will work, that a callable is passed
        # TODO: Check that pipeline_f is callable
        else:
            self.logger.info(f"{str(self.pre_processing_pipeline)} pipeline passed directly")
            self.pipeline_f = self.pre_processing_pipeline

        if isinstance(self.pipeline_f, Pipeline):
            self.pipeline_obj = self.pipeline_f
            self.pipeline_f = self.pipeline_obj.transform

        # If no data sharing, then load and parse data from scratch
        if self.data_from is None:
            self.logger.info("Loading data directly")
            # Leave this here for now...
            # self.mfcc_m = torchaudio.transforms.MFCC(self.default_audio_sample_rate,
            #                                         self.num_mfcc)

            ## Data loading ##
            # - Load the data, parsing into pandas data frame/series types
            # - Only minimal processing into Python objects done here
            data_iter = tqdm(self.patient_tuples, desc="Loading data")
            mat_data_maps = {l_p_s_t_tuple: self.load_data(*l_p_s_t_tuple,
                                                           # sensor_columns=self.sensor_columns,
                                                           # IMPORTANT: Don't parse data yet
                                                           # parse_mat_data=False,
                                                           subset=self.data_subset)
                             for l_p_s_t_tuple in data_iter}

            #  Important processing  #
            # - Process each subject in data map through pipeline func
            self.sample_index_maps = dict()
            self.data_maps = dict()

            for k, dmap in mat_data_maps.items():
                # Run the pipeline, mutating/modifying the data map for this patient trial
                res_dmap = self.pipeline_f(dmap)
                self.sample_index_maps[k] = res_dmap['sample_index_map']
                # THe first data map sets the sampling frequency fs
                # self.fs_signal = getattr(self, 'fs_signal', res_dmap[self.mat_d_keys['signal_fs']])
                self.fs_signal = res_dmap[self.mat_d_keys['signal_fs']] if self.fs_signal is None else self.fs_signal

                # self.ecog_window_size = getattr(self, 'ecog_window_size',
                #                                int(self.fs_signal * self.sample_ixer.window_size.total_seconds()))
                # self.ecog_window_size = int(self.fs_signal * self.sample_ixer.window_size.total_seconds())

                # self.n_samples_per_window = res_dmap['n_samples_per_window']
                self.n_samples_per_window = getattr(self, 'n_samples_per_window', res_dmap['n_samples_per_window'])
                self.logger.info(f"N samples per window: {self.n_samples_per_window}")

                if self.fs_signal != res_dmap[self.mat_d_keys['signal_fs']]:
                    raise ValueError("Mismatch fs (%s!=%s) on %s" % (self.fs_signal, res_dmap['fs_signal'], str(k)))

                self.data_maps[k] = res_dmap

            ###
            # Sensor selection logic - based on the patients loaded - which sensors do we use?
            if self.sensor_columns is None or isinstance(self.sensor_columns, str):
                # good_and_bad_tuple_d = {l_p_s_t_tuple: self.identify_good_and_bad_sensors(mat_d, self.sensor_columns)
                #                            for l_p_s_t_tuple, mat_d in mat_data_maps.items()}
                good_and_bad_tuple_d = {l_p_s_t_tuple: (mat_d['good_sensor_columns'], mat_d['bad_sensor_columns'])
                                        for l_p_s_t_tuple, mat_d in self.data_maps.items()}

                good_and_bad_tuple_d = {
                    k: (set(gs) if gs else (list(range(self.data_maps[k][self.mat_d_keys['signal']].shape[1]))),
                        bs)
                    for k, (gs, bs) in good_and_bad_tuple_d.items()}

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

                # elif self.sensor_columns == 'all':
                else:
                    raise ValueError("Unknown snsor columns argument: " + str(self.sensor_columns))
                # print("Selected columns with -%s- method: %s"
                #      % (self.sensor_columns, ", ".join(map(str, self.selected_columns))) )
                self.logger.info(f"Selected {len(self.selected_columns)} columns using {self.sensor_columns} method: "
                                 f"{', '.join(map(str, self.selected_columns))}")
                #self.logger.info("Selected columns with -%s- method: %s"
                #                 % (self.sensor_columns, ", ".join(map(str, self.selected_columns))))
            else:
                self.selected_columns = self.sensor_columns

            self.sensor_count = len(self.selected_columns)

            #            if self.flatten_sensors_to_samples:
            #                # Map full description (word label, window index, sensor,...trial key elements..}
            #                # to the actual pandas index
            #                self.logger.info("Flattening sensors to samples")
            #                self.flat_index_map = {tuple([wrd_id, ix_i, s_i] + list(k_t)): ixes
            #                                       for k_t, index_map in tqdm(self.sample_index_maps.items(),
            #                                                                  desc="Creating sample indices")
            #                                       for wrd_id, ix_list in index_map.items()
            #                                       for ix_i, ixes in enumerate(ix_list)
            #                                       for s_i, s in enumerate(self.selected_columns)}
            #
            #                # Enumerate all the keys across flat_index_map into one large list for index-style,
            #                # has a len() and can be indexed into nicely (via np.ndarray)
            #                self.flat_keys = np.array([(k, k[3:])
            #                                           for i, k in enumerate(self.flat_index_map.keys())],
            #                                          dtype='object')
            #            else:
            #                # Map full description (word label, window index,...trial key elements..}
            #                # to the actual pandas index
            #                self.flat_index_map = {tuple([wrd_id, ix_i] + list(k_t)): ixes
            #                                       for k_t, index_map in tqdm(self.sample_index_maps.items(),
            #                                                                  desc="Creating sample indices")
            #                                       for wrd_id, ix_list in index_map.items()
            #                                       for ix_i, ixes in enumerate(ix_list)}
            #
            #                # Enumerate all the keys across flat_index_map into one large list for index-style,
            #                # has a len() and can be indexed into nicely (via np.ndarray)
            #                self.flat_keys = np.array([(k, k[2:])
            #                                           for i, k in enumerate(self.flat_index_map.keys())],
            #                                          dtype='object')

            # ### New Version ###
            sample_ix_df_l = list()
            key_cols = ['label', 'sample_ix', 'location', 'patient', 'session', 'trial']
            for l_p_s_t, index_map in self.sample_index_maps.items():
                patient_ixes = list()

                cols = key_cols + ['start_t', 'stop_t', 'indices']

                key_l = list(l_p_s_t)

                patient_ixes += [tuple([label_code, ix_i] + key_l + [_ix.min(), _ix.max(), _ix])
                                 for label_code, indices_l in index_map.items()
                                 for ix_i, _ix in enumerate(indices_l)]

                p_ix_df = pd.DataFrame(patient_ixes, columns=cols)

                if 'word_start_stop_times' in self.data_maps[l_p_s_t]:
                    wsst_df = self.data_maps[l_p_s_t]['word_start_stop_times']
                    nearest_ixes = wsst_df.index.get_indexer(p_ix_df.start_t, method='nearest')
                    p_ix_df['sent_code'] = wsst_df.iloc[nearest_ixes].stim_sentcode.values

                sample_ix_df_l.append(p_ix_df)

            self.sample_ix_df = pd.concat(sample_ix_df_l).reset_index(drop=True)
            k_select_offset = 2
            if self.flatten_sensors_to_samples:
                self.sample_ix_df['channel'] = [self.selected_columns] * len(self.sample_ix_df)
                key_cols.insert(2, 'channel')
                k_select_offset += 1
                self.sample_ix_df = self.sample_ix_df.explode('channel')

            self.key_cols = key_cols

            key_df = self.sample_ix_df[self.key_cols]
            self.flat_keys = np.array(list(zip(key_df.to_records(index=False).tolist(),
                                               key_df.iloc[:, k_select_offset:].to_records(index=False).tolist())),
                                      dtype='object')
            self.flat_index_map = self.sample_ix_df.set_index(key_cols).indices.to_dict()

            # ## END NEW VERSION

            self.logger.info(f"Length of flat index map: {len(self.flat_index_map)}")

            # Finish processing the data mapping loaded from the mat data files
            # self.data_maps = {l_p_s_t_tuple: self.parse_mat_arr_dict(mat_d, self.selected_columns)
            #                  for l_p_s_t_tuple, mat_d in tqdm(mat_data_maps.items(), desc='Parsing data')}
            ###-----
            assert self.sensor_count == len(self.selected_columns)
            # print(f"Selected {len(self.selected_columns)} sensors")
            self.logger.info(f"Selected {len(self.selected_columns)} sensors")
            ###-----

            self.sensor_selection_trf = pipeline.ApplySensorSelection(selection=self.selected_columns)
            self.data_maps = {l_p_s_t_tuple: self.sensor_selection_trf.transform(mat_d)
                              for l_p_s_t_tuple, mat_d in tqdm(self.data_maps.items(),
                                                               desc='Applying sensor selection')}

        else:
            # print("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            self.logger.warning("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            # self.mfcc_m = self.data_from.mfcc_m
            self.data_maps = self.data_from.data_maps
            self.sample_index_maps = self.data_from.sample_index_maps
            self.flat_index_map = self.data_from.flat_index_map
            self.flat_keys = self.data_from.flat_keys
            self.key_cols = self.data_from.key_cols
            # self.logger.info("Copying over sample ix dataframe")
            self.sample_ix_df = self.data_from.sample_ix_df.copy()
            self.selected_columns = self.data_from.selected_columns
            self.flatten_sensors_to_samples = self.data_from.flatten_sensors_to_samples
            self.extra_output_keys = self.data_from.extra_output_keys
            self.fs_signal = self.data_from.fs_signal

        self.select(self.selected_word_indices)

    def make_pipeline_map(self, default='audio_gate'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """
        self.logger.debug(f"default pipeline: {default}")
        p_map = {
            'audio_gate': Pipeline([
                ('parse_signal', pipeline.ParseTimeSeriesArrToFrame(self.mat_d_keys['signal'],
                                                                    self.mat_d_keys['signal_fs'],
                                                                    default_fs=1200, output_key='signal')),
                ('parse_audio', pipeline.ParseTimeSeriesArrToFrame(self.mat_d_keys['audio'],
                                                                   self.mat_d_keys['audio_fs'],
                                                                   default_fs=48000, reshape=-1)),
                ('parse_stim', pipeline.ParseTimeSeriesArrToFrame(self.mat_d_keys['stimcode'],
                                                                  self.mat_d_keys['signal_fs'],
                                                                  default_fs=1200, reshape=-1, output_key='stim')),
                ('sensor_selection', pipeline.IdentifyGoodAndBadSensors(sensor_selection=self.sensor_columns)),
                ('subsample', pipeline.SubsampleSignal()),
                ('Threshold', pipeline.PowerThreshold(speaking_window_samples=48000 // 16,
                                                      silence_window_samples=int(48000 * 1.5),
                                                      speaking_quantile_threshold=0.9,
                                                      # n_silence_windows=5000,
                                                      # silence_threshold=0.001,
                                                      # silGence_quantile_threshold=0.05,
                                                      silence_n_smallest=5000)),
                ('speaking_indices', pipeline.WindowSampleIndicesFromStim('stim_pwrt',
                                                                          target_onset_shift=pd.Timedelta(-.5, 's'),
                                                                          # input are centers, and output is a window of .5 sec
                                                                          # so to center it, move the point (center) back .25 secods
                                                                          # so that extracted 0.5 sec window saddles the original center
                                                                          # target_offset_shift=pd.Timedelta(-0.25, 's')
                                                                          target_offset_shift=pd.Timedelta(-0.5, 's')
                                                                          )
                 ),

                ('silence_indices', pipeline.WindowSampleIndicesFromIndex('silence_stim_pwrt_s',
                                                                          # Center the extracted 0.5 second window
                                                                          index_shift=pd.Timedelta(-0.25, 's'),
                                                                          stim_value_remap=0
                                                                          )),
                ('output', 'passthrough')
            ]),

            #'minimal':
            #    feature_processing.SubsampleECOG() >>
            #    feature_processing.WordStopStartTimeMap() >> feature_processing.ChangSampleIndicesFromStim()
        }
        p_map['default'] = p_map[default]

        return p_map

    def to_eval_replay_dataloader(self, patient_k=None, data_k='ecog', stim_k='stim', win_step=1, batch_size=1024,
                                  num_workers=4,
                                  ecog_transform=None):
        if patient_k is None:
            patient_k = list(self.data_maps.keys())
        elif not isinstance(patient_k, list):
            patient_k = [patient_k]

        dl_map = dict()
        for k in patient_k:
            data_map = self.data_maps[k]
            ecog_torch_arr = torch.from_numpy(data_map[data_k].values).float()
            outputs = list()
            for _iix in tqdm(range(0, ecog_torch_arr.shape[0] - self.ecog_window_size, win_step),
                             desc='creating windows'):
                _ix = slice(_iix, _iix + self.ecog_window_size)
                feats = self.get_features(data_map, _ix, transform=ecog_transform, index_loc=True)
                # TODO: Just grabbing the max stim wode in the range - better or more useful way to do this?
                targets = self.get_targets(data_map, None, label=data_map['stim'].iloc[_ix].max())
                so = dict(**feats, **targets)
                so = {k: v for k, v in so.items()
                      if isinstance(v, torch.Tensor)}
                outputs.append(so)
            t_dl = torch.utils.data.DataLoader(outputs, batch_size=batch_size, num_workers=num_workers)
            dl_map[k] = t_dl

        ret = dl_map
        # if len(ret) == 1:
        #    ret = list(dl_map.values())[0]
        return ret

    def __len__(self):
        return len(self.selected_flat_keys)

    def __getitem__(self, item):
        # ix_k includes the class and window id, and possibly sensor id if flattened
        # data_k specifies subject dataset in data_map (less granular than ix_k)
        ix_k, data_k = self.selected_flat_keys[item]
        data_d = self.data_maps[data_k]

        selected_channels = None
        if self.flatten_sensors_to_samples:
            selected_channels = [ix_k[2]]

        so = dict()

        so.update(
            self.get_features(data_d, self.flat_index_map[ix_k],
                              ix_k, transform=self.transform,
                              channel_select=selected_channels,
                              extra_output_keys=self.extra_output_keys)
        )

        so.update(
            self.get_targets(data_d, self.flat_index_map[ix_k], ix_k, target_transform=self.target_transform)
        )

        if self.post_processing_func is not None:
            so_updates = self.post_processing_func(so, **self.post_processing_kws)
            so.update(so_updates)

        # Return anything that is a Torch Tensor - the torch dataloader will handle
        # compiling multiple outputs for batch
        return {k: v for k, v in so.items()
                if isinstance(v, torch.Tensor)}

    def split_select_at_time(self, split_time: float):
        # split_time = 0.75
        from tqdm.auto import tqdm

        selected_keys_arr = self.flat_keys[self.selected_word_indices]
        index_start_stop = [(self.flat_index_map[a[0]].min(), self.flat_index_map[a[0]].max())
                            for a in tqdm(selected_keys_arr)]
        split_time = max(a for a, b, in index_start_stop) * split_time if isinstance(split_time, float) else split_time
        left_side_indices, right_side_indices = list(), list()
        for a, b in index_start_stop:
            if a < split_time:
                left_side_indices.append(a)
            else:
                right_side_indices.append(b)

        left_side_indices = np.array(left_side_indices)
        right_side_indices = np.array(right_side_indices)

        left_dataset = self.__class__(data_from=self, selected_word_indices=left_side_indices)
        right_dataset = self.__class__(data_from=self, selected_word_indices=right_side_indices)

        return left_dataset, right_dataset

    def split_select_random_key_levels(self, key='sent_code', **train_test_split_kws):
        from sklearn.model_selection import train_test_split
        s: pd.Series = self.sample_ix_df[key]
        levels = s.unique()
        train, test = train_test_split(levels, **train_test_split_kws)
        self.logger.info(f"{len(levels)} levels in {key} split into train/test")
        self.logger.info(f"Train: {train}")
        self.logger.info(f"Test : {test}")

        train_mask = s.isin(train)
        test_mask = s.isin(test)

        train_indices = self.sample_ix_df[train_mask].index.tolist()
        test_indices = self.sample_ix_df[test_mask].index.tolist()

        train_dataset = self.__class__(data_from=self, selected_word_indices=train_indices)
        test_dataset = self.__class__(data_from=self, selected_word_indices=test_indices)

        return train_dataset, test_dataset

    def select(self, sample_indices):
        # select out specific samples from the flat_keys array if selection passed
        # - Useful if doing one-subject training and want to split data up among datasets for use
        self.selected_word_indices = sample_indices
        if self.selected_word_indices is not None:
            self.selected_flat_keys = self.flat_keys[self.selected_word_indices]
        else:
            self.selected_flat_keys = self.flat_keys

        return self

    def append_transform(self, transform):
        transform = [transform] if not isinstance(transform, list) else transform
        self.transform_l += transform
        self.transform = torchvision.transforms.Compose(self.transform_l)
        return self

    def append_target_transform(self, transform):
        self.target_transform_l.append(transform)
        self.target_transform = torchvision.transforms.Compose(self.target_transform_l)
        return self

    ######
    @classmethod
    def load_mat_keys_from_path(cls, p):
        """
        Returns only the keys in the HDF5 file without loading the data
        """
        import h5py
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
            msg = f"Couldn't load {os.path.split(p)[-1]} with scipy (vers > 7.3?) - using package 'mat73' to load"
            cls.logger.info(msg)

            import mat73
            mat_dat_map = mat73.loadmat(p)
        return mat_dat_map

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
                  sensor_columns=None, subset=None):

        location = cls.default_location if location is None else location
        patient = cls.default_patient if patient is None else patient
        session = cls.default_session if session is None else session
        trial = cls.default_trial if trial is None else trial
        sensor_columns = cls.default_sensor_columns if sensor_columns is None else sensor_columns

        cls.logger.info(f"---{patient}-{session}-{trial}-{location}---")
        cls.logger.info("|--->Using Subset: " + str(subset))

        p = cls.get_data_path(patient, session, trial, location, base_path=base_path, subset=subset)
        cls.logger.debug(f"Path : {p}")

        mat_d = cls.load_mat_from_path(p)
        cls.logger.debug(f"Matlab keys : {list(mat_d.keys())}")

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

        if '~' == sets_str[0]:
            return cls.make_all_tuples_with_one_left_out(sets_str[1:])

        set_terms = sets_str.split('-')
        # e.g. MC-22-1 has three terms ('MC', 22, 1) selecting a specific trial of a specific participant
        if len(set_terms) == 3:
            # org, pid, ix = sets_str.split('-')
            org, pid, ix = set_terms
            assert pid.isdigit() and ix.isdigit() and org in cls.all_patient_maps.keys()
            pmap, pid, ix = cls.all_patient_maps[org], int(pid), int(ix)
            assert pid in pmap, f"PID: {pid} not in {org}'s known data map"
            p_list = [pmap[pid][ix]]
        # e.g. MC-22 will return tuples for all of MC-22's data
        elif len(set_terms) == 2:
            org, pid = set_terms

            assert pid.isdigit(), f"pid expected to be a digit, but got {pid}"
            assert org in cls.all_patient_maps.keys(), f"org expected to be one of {list(cls.all_patient_maps.keys())}, but got {org}"

            pmap, pid = cls.all_patient_maps[org], int(pid)
            assert pid in pmap, f"PID: {pid} not in {org}'s known data map"
            p_list = pmap[pid]
        else:
            raise ValueError(f"Don't understand the {len(set_terms)} terms: {set_terms}")

        return p_list

    @classmethod
    def make_all_tuples_with_one_left_out(cls, sets_str):
        selected_t_l = cls.make_tuples_from_sets_str(sets_str)
        remaining_t_l = sum((l for pid_to_t_l in cls.all_patient_maps.values() for l in pid_to_t_l.values() if
                             all(o not in selected_t_l for o in l)),
                            start=list())
        return remaining_t_l

    @staticmethod
    def get_features(data_map, ix, label=None, transform=None, index_loc=False, signal_key='signal',
                     channel_select=None, extra_output_keys=None):
        # pull out signal and begin building dictionary of arrays to reutrn
        signal_df = data_map[signal_key]

        kws = dict()
        kws['signal'] = signal_df.loc[ix] if not index_loc else signal_df.iloc[ix]

        # Transpose to keep time as last index for torch
        np_ecog_arr = kws['signal'].values.T

        # if self.flatten_sensors_to_samples:
        # Always pass a list/array for channels, even if only 1, to maintain the dimension
        if channel_select is not None:
            np_ecog_arr = np_ecog_arr[channel_select]  # [None, :]

        if transform is not None:
            # print("Apply transform to shape of " + str(np_ecog_arr.shape))
            np_ecog_arr = transform(np_ecog_arr)

        kws['signal_arr'] = torch.from_numpy(np_ecog_arr).float()

        # extra_output_keys = ['sensor_ras_coord_arr']
        extra_output_keys = [extra_output_keys] if isinstance(extra_output_keys, str) else extra_output_keys
        if isinstance(extra_output_keys, list):
            kws.update({k: torch.from_numpy(data_map[k]).float() if isinstance(data_map[k], np.ndarray) else data_map[k]
                        for k in extra_output_keys})

            if 'sensor_ras_coord_arr' in kws and channel_select is not None:
                #                print(channel_select)
                #                if not isinstance(channel_select[0], int):
                #                    print("WHAT")
                kws['sensor_ras_coord_arr'] = kws['sensor_ras_coord_arr'][channel_select].unsqueeze(0)

        return kws

    def get_feature_shape(self):
        # TODO: Don't hardode signal array everywhere
        return self[0]['signal_arr'].shape

    @staticmethod
    def get_targets(data_map, ix, label, target_transform=None, target_key='target_arr'):
        label = label[0]
        #kws = dict(text='<silence>' if label <= 0 else '<speech>',
        #           text_arr=torch.Tensor([0] if label <= 0 else [1]))
        kws = {target_key: torch.LongTensor([label])}
        if target_transform is not None:
            kws[target_key] = target_transform(kws[target_key])
        return kws

    @staticmethod
    def get_targets_old(data_map, ix, label, target_transform=None):
        label = label[0]

        kws = dict(text='<silence>' if label <= 0 else '<speech>',
                   text_arr=torch.Tensor([0] if label <= 0 else [1]))
        if target_transform is not None:
            kws['text_arr'] = target_transform(kws['text_arr'])
        return kws

    def sample_plot(self, i, band=None,
                    offset_seconds=0,
                    figsize=(15, 10), axs=None):
        import matplotlib
        from matplotlib import pyplot as plt
        from IPython.display import display
        # offs = pd.Timedelta(offset_seconds)
        # t_word_ix = self.word_index[self.word_index == i].index
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
        word_txt = "couldn't get word or text mapping from data_map"
        if 'word_code_d' in self.data_maps[data_k]:
            word_txt = self.data_maps[data_k]['word_code_d'].get(ix_k[0], '<no speech>')

        t_word_ecog_df = ecog_df.loc[t_word_slice].dropna()
        t_word_wav_df = speech_df.loc[t_word_slice]
        # display(t_word_ecog_df.describe())
        # scols = self.default_sensor_columns
        scols = self.selected_columns

        ecog_std = ecog_df[scols].std()
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=ecog_std.min(), vmax=ecog_std.max())
        c_f = lambda v: cmap(norm(v))
        colors = ecog_std.map(c_f).values

        if axs is None:
            fig, axs = plt.subplots(figsize=figsize, nrows=2)
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

    def get_target_shape(self):#, target_key='target_arr'):
        return self.sample_ix_df.label.nunique()


@attr.s
@with_logger
class HarvardSentences(BaseASPEN):
    """
    """

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
        4: [('UCSD', 4, 1, 1)],
        5: [('UCSD', 5, 1, 1)],
        10: [('UCSD', 10, 1, 1)],
        18: [('UCSD', 18, 1, 1)],
        19: [('UCSD', 19, 1, 1)],
        22: [('UCSD', 22, 1, 1)],
        28: [('UCSD', 28, 1, 1)],
    })

    def make_pipeline_map(self, default='audio_gate'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """
        parse_arr_steps = [
            ('parse_signal', pipeline.ParseTimeSeriesArrToFrame(self.mat_d_keys['signal'],
                                                                self.mat_d_keys['signal_fs'],
                                                                1200, output_key='signal')),
            ('parse_audio', pipeline.ParseTimeSeriesArrToFrame(self.mat_d_keys['audio'],
                                                               self.mat_d_keys['audio_fs'],
                                                               48000, reshape=-1)),
            ('parse_stim', pipeline.ParseTimeSeriesArrToFrame(self.mat_d_keys['stimcode'],
                                                              self.mat_d_keys['signal_fs'],
                                                              1200, reshape=-1, output_key='stim')),
            ('parse_sensor_ras', pipeline.ParseSensorRAS()),
            ('extract_mfc', pipeline.ExtractMFCC())
        ]

        parse_input_steps = [
            ('sensor_selection', pipeline.IdentifyGoodAndBadSensors(sensor_selection=self.sensor_columns)),
            ('rescale_signal', pipeline.StandardNormSignal()),
            ('subsample', pipeline.SubsampleSignal()),
            ('sent_from_start_stop', pipeline.SentCodeFromStartStopWordTimes()),

        ]

        #parse_stim_steps = [
        #    # Produces imagine_start/stop_t and mouth_start/stop_t
        #    ('stim_from_start_stop', pipeline.SentenceAndWordStimFromRegionStartStopTimes()),
        #]


        audio_gate_steps = [
            ('Threshold', pipeline.PowerThreshold(speaking_window_samples=48000 // 16,
                                                  silence_window_samples=int(48000 * 1.5),
                                                  speaking_quantile_threshold=0.9,
                                                  # silence_threshold=0.001,
                                                  silence_quantile_threshold=0.05,
                                                  n_silence_windows=35000,
                                                  # silence_n_smallest=30000,
                                                  #stim_key='speaking_region_stim'
                                                  stim_key='speaking_region_stim_mask'
                                                  )),
            ('speaking_indices', pipeline.WindowSampleIndicesFromStim('stim_pwrt',
                                                                      target_onset_shift=pd.Timedelta(-.5, 's'),
                                                                      # input are centers, and output is a window of
                                                                      # .5 sec so to center it, move the point (
                                                                      # center) back .25 secods so that extracted 0.5
                                                                      # sec window saddles the original center
                                                                      # target_offset_shift=pd.Timedelta(-0.25, 's')
                                                                      target_offset_shift=pd.Timedelta(-0.5, 's'),
                                                                      #max_target_region_size=300
                                                                      sample_n=1000,
                                                                      )),
            ('silence_indices', pipeline.WindowSampleIndicesFromIndex('silence_stim_pwrt_s',
                                                                      # Center the extracted 0.5 second window
                                                                      index_shift=pd.Timedelta(-0.25, 's'),
                                                                      stim_value_remap=0,
                                                                      sample_n=10000,
                                                                      ))]

        start_stop_steps = [('new_mtss', pipeline.NewNewMultiTaskStartStop()),
                                                 # Stims from Start-stop-times
                                                 ('speaking_word_stim', pipeline.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='start_t',
                                                                        stop_t_column='stop_t',
                                                                        stim_output_name='speaking_word_stim',
                                                 )),
                                                 ('listening_word_stim', pipeline.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='listening_word_start_t',
                                                                        stop_t_column='listening_word_stop_t',
                                                                        stim_output_name='listening_word_stim',
                                                 )),
                                                 ('mouthing_word_stim', pipeline.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='mouthing_word_start_t',
                                                                        stop_t_column='mouthing_word_stop_t',
                                                                        stim_output_name='mouthing_word_stim',
                                                 )),
                                                ('imagining_word_stim', pipeline.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='imagining_word_start_t',
                                                                        stop_t_column='imagining_word_stop_t',
                                                                        stim_output_name='imagining_word_stim',
                                                )),

                            ('speaking_region_stim', pipeline.NewStimFromRegionStartStopTimes(
                                start_t_column='speaking_region_start_t',
                                stop_t_column='speaking_region_stop_t',
                                stim_output_name='speaking_region_stim',
                            )),
                            ('listening_region_stim', pipeline.NewStimFromRegionStartStopTimes(
                                start_t_column='listening_region_start_t',
                                stop_t_column='listening_region_stop_t',
                                stim_output_name='listening_region_stim',
                            )),
                            ('mouthing_region_stim', pipeline.NewStimFromRegionStartStopTimes(
                                start_t_column='mouthing_region_start_t',
                                stop_t_column='mouthing_region_stop_t',
                                stim_output_name='mouthing_region_stim',
                            )),
                            ('imagining_region_stim', pipeline.NewStimFromRegionStartStopTimes(
                                start_t_column='imagining_region_start_t',
                                stop_t_column='imagining_region_stop_t',
                                stim_output_name='imagining_region_stim',
                            ))
                            ]


        region_kws = dict(
            target_onset_shift=pd.Timedelta(0.1, 's'),
            target_offset_shift=pd.Timedelta(-0.1, 's'),
            sample_n=1000
            #max_target_region_size=1200
        )
        region_from_word_kws = dict(
            target_onset_shift=pd.Timedelta(-.5, 's'),
            target_offset_shift=pd.Timedelta(-0.5, 's'),
        )
        p_map = {
            'random_sample': Pipeline(parse_arr_steps + parse_input_steps
            + [('rnd_stim', pipeline.RandomStim(10_000)),
               ('rnd_indices', pipeline.WindowSampleIndicesFromIndex(stim_key='random_stim'))]
                                      + [('output', 'passthrough')]),

            # -----
            # Directly from audio
            'audio_gate': Pipeline(parse_arr_steps + parse_input_steps  + start_stop_steps  #+ parse_stim_steps
                                   + audio_gate_steps + [('output', 'passthrough')]),

            'region_classification': Pipeline(parse_arr_steps + parse_input_steps + start_stop_steps
                                                             + [
                                                                 # Indices from Stim - these populate the class labels
                                                                 ('speaking_indices',
                                                                  pipeline.WindowSampleIndicesFromStim(
                                                                      'speaking_region_stim',
                                                                      stim_value_remap=0, **region_kws)),
                                                                 ('listening_indices',
                                                                  pipeline.WindowSampleIndicesFromStim(
                                                                      'listening_region_stim',
                                                                      stim_value_remap=1, **region_kws)),
                                                                 ('mouthing_indices',
                                                                  pipeline.WindowSampleIndicesFromStim(
                                                                      'mouthing_region_stim',
                                                                      stim_value_remap=2, **region_kws)),
                                                                 ('imagining_indices',
                                                                  pipeline.WindowSampleIndicesFromStim(
                                                                      'imagining_region_stim',
                                                                      stim_value_remap=3, **region_kws)),
                                                                 ('output', 'passthrough')
                                                             ]),

            'region_classification_from_word_stim': Pipeline(parse_arr_steps + parse_input_steps + start_stop_steps
                                              + [
                                                # Indices from Stim - these populate the class labels
                                                ('speaking_indices', pipeline.WindowSampleIndicesFromStim(
                                                    'speaking_word_stim',
                                                    stim_value_remap=0,
                                                    **region_from_word_kws
                                                )),
                                                 ('listening_indices', pipeline.WindowSampleIndicesFromStim(
                                                    'listening_word_stim',
                                                    stim_value_remap=1,
                                                    **region_from_word_kws
                                                 )),
                                                 ('mouthing_indices', pipeline.WindowSampleIndicesFromStim(
                                                    'mouthing_word_stim',
                                                    stim_value_remap=2,
                                                    **region_from_word_kws

                                                 )),
                                                 ('imagining_indices', pipeline.WindowSampleIndicesFromStim(
                                                    'imagining_word_stim',
                                                    stim_value_remap=3,
                                                    **region_from_word_kws
                                                 )),('output', 'passthrough')]),

            'audio_gate_speaking_only': Pipeline(parse_arr_steps + parse_input_steps  + start_stop_steps
                                                 # Slice out the generation of the silence stim data - only speaking
                                                 + audio_gate_steps[:-1] + [('output', 'passthrough')]),

#            'audio_gate_imagine': Pipeline(parse_arr_steps + parse_input_steps + [
#                # Creates listening, imagine, mouth
#                #('multi_task_start_stop', pipeline.MultiTaskStartStop()),
#                # Creates the word_stim and sentence_stim from the start stop of imagine
#                ('stim_from_start_stop', pipeline.SentenceAndWordStimFromRegionStartStopTimes(start_t_column='imagine_start_t',
#                                                                                              stop_t_column='imagine_stop_t')),
#                # creat stim for listening (i.e. not speaking or active) that we'll use for silent
#                ('stim_from_listening', pipeline.SentenceAndWordStimFromRegionStartStopTimes(start_t_column='listening_region_start_t',
#                                                                                             stop_t_column='listening_region_stop_t',
#                                                                                             word_stim_output_name='listening_word_stim',
#                                                                                             sentence_stim_output_name='listening_sentence_stim',
#                                                                                             set_as_word_stim=False)),
#                # Target index extraction - word stim is the imagine stim extracted above
#                ('speaking_indices', pipeline.WindowSampleIndicesFromStim('word_stim',
#                                                                          target_onset_shift=pd.Timedelta(-.5, 's'),
#                                                                          target_offset_shift=pd.Timedelta(-0.5, 's'),
#                                                                          )),
#                # Negative target index extraction - use listening regions for negatives
#                ('silent_indices', pipeline.WindowSampleIndicesFromStim('listening_word_stim',
#                                                                        target_onset_shift=pd.Timedelta(.5, 's'),
#                                                                        target_offset_shift=pd.Timedelta(-0.5, 's'),
#                                                                        stim_value_remap=0,
#                                                                        )),
#
#                ('output', 'passthrough')
#            ]),

        }

        p_map['default'] = p_map[default]

        return p_map

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
class HarvardSentencesMFC(HarvardSentences):

    @staticmethod
    def get_targets(data_map, ix, label, target_transform=None):
        mel_ix = data_map['audio_mel_spec'].index.get_indexer(ix, method='nearest')

        mel_spec_df = data_map['audio_mel_spec'].iloc[mel_ix]
        mel_spec_arr = torch.from_numpy(mel_spec_df.values.T)

        # Just window E() - should this be last sample? Center sample? Median?
        mel_spec_arr = mel_spec_arr.mean(-1)

        return dict(target=mel_spec_arr)

    def get_target_shape(self):
        return self.pipeline_obj.named_steps.extract_mfc.n_mels


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

        # 25: [('MayoClinic', 25, 1, 1),
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
    tuple_to_sets_str_map = {t: f"{l}-{p}-{i}"
                             for l, p_d in all_patient_maps.items()
                             for p, t_l in p_d.items()
                             for i, t in enumerate(t_l)}

    #######
    ## Path handling
    @classmethod
    def make_filename(cls, patient, session, trial, location):
        if location in cls.fname_prefix_map:  # == 'Mayo Clinic':
            return f"{cls.fname_prefix_map.get(location)}{str(patient).zfill(3)}-SW-S{session}-R{trial}.mat"
        else:
            raise ValueError("Don't know location " + location)


@attr.s
class ChangNWW(NorthwesternWords):
    """
    Northwester-style with Chang pre-processing steps
    """
    # data_subset = 'Preprocessed/Chang1'
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

    def make_pipeline_map(self, default='audio_gate'):
        raise NotImplementedError("ChangNWW with MFC pipeline components not in new skl framework")


# ###########
# Options ---
@dataclass
class DatasetOptions(JsonSerializable):
    dataset_name: str = None

    batch_size: int = 256
    batches_per_epoch: Optional[int] = None
    """If set, only does this many batches in an epoch - otherwise, will do enough batches to equal dataset size"""

    pre_processing_pipeline: str = 'default'

    train_sets: str = None
    cv_sets: Optional[str] = None
    test_sets: Optional[str] = None

    data_subset: str = 'Data'
    output_key: str = 'signal_arr'
    extra_output_keys: Optional[str] = None
    random_sensors_to_samples: bool = False
    flatten_sensors_to_samples: bool = False
    # power_q: float = 0.7
    random_targets: bool = False

    n_dl_workers: int = 4
    n_dl_eval_workers: int = 6

    def make_datasets_and_loaders(self, dataset_cls=None, base_data_kws=None,
                                  train_data_kws=None, cv_data_kws=None, test_data_kws=None,
                                  train_sets_str=None, cv_sets_str=None, test_sets_str=None,
                                  train_sensor_columns='valid',
                                  pre_processing_pipeline=None,
                                  additional_transforms=None,
                                  # additional_train_transforms=None, additional_eval_transforms=None,
                                  #num_dl_workers=None
                                  ) -> tuple:
        """
        Helper method to create instances of dataset_cls as specified in the command-line options and
        additional keyword args.
        Parameters
        ----------
        options: object
            Options object build using the utils module
        dataset_cls: Derived class of BaseDataset (default=None)
            E.g. NorthwesterWords
        train_data_kws: dict (default=None)
            keyword args to train version of the dataset
        cv_data_kws: dict (default=None)
            keyword args to cv version of the dataset
        test_data_kws: dict (default=None)
            keyword args to test version of the dataset
        num_dl_workers: int (default=8)
            Number of workers in each dataloader. Can be I/O bound, so sometimes okay to over-provision

        Returns
        -------
        dataset_map, dataloader_map, eval_dataloader_map
            three-tuple of (1) map to original dataset (2) map to the constructed dataloaders and
            (3) Similar to two, but not shuffled and larger batch size (for evaluation)
        """
        # from torchvision import transforms
        # raise ValueError("BREAK")
        base_data_kws = dict() if base_data_kws is None else base_data_kws
        if dataset_cls is None:
            dataset_cls = BaseDataset.get_dataset_by_name(self.dataset_name)

        train_p_tuples = dataset_cls.make_tuples_from_sets_str(self.train_sets if train_sets_str is None
                                                               else train_sets_str)
        cv_p_tuples = dataset_cls.make_tuples_from_sets_str(self.cv_sets if cv_sets_str is None
                                                            else cv_sets_str)
        test_p_tuples = dataset_cls.make_tuples_from_sets_str(self.test_sets if test_sets_str is None
                                                              else test_sets_str)
        logger.info("Train tuples: " + str(train_p_tuples))
        logger.info("CV tuples: " + str(cv_p_tuples))
        logger.info("Test tuples: " + str(test_p_tuples))

        base_kws = dict(pre_processing_pipeline=self.pre_processing_pipeline
        if pre_processing_pipeline is None
        else pre_processing_pipeline,
                        data_subset=self.data_subset,
                        extra_output_keys=self.extra_output_keys.split(
                            ',') if self.extra_output_keys is not None else None,
                        flatten_sensors_to_samples=self.flatten_sensors_to_samples)

        base_kws.update(base_data_kws)
        logger.info(f"Dataset base keyword arguments: {base_kws}")
        train_kws = dict(patient_tuples=train_p_tuples, **base_kws)
        cv_kws = dict(patient_tuples=cv_p_tuples, **base_kws)
        test_kws = dict(patient_tuples=test_p_tuples, **base_kws)

        if train_data_kws is not None:
            train_kws.update(train_data_kws)
        if cv_data_kws is not None:
            cv_kws.update(cv_data_kws)
        if test_data_kws is not None:
            test_kws.update(test_data_kws)

        dl_kws = dict(num_workers=self.n_dl_workers, batch_size=self.batch_size,
                      batches_per_epoch=self.batches_per_epoch,
                      shuffle=False, random_sample=True)
        print(f"DL Keyword arguments: {dl_kws}")
        eval_dl_kws = dict(num_workers=self.n_dl_eval_workers, batch_size=512,
                           batches_per_epoch=self.batches_per_epoch,
                           shuffle=False, random_sample=False if self.batches_per_epoch is None else True)

        dataset_map = dict()
        logger.info("Using dataset class: %s" % str(dataset_cls))
        # Setup train dataset
        train_nww = dataset_cls(sensor_columns=train_sensor_columns, **train_kws)

        roll_channels = getattr(self, 'roll_channels', False)
        shuffle_channels = getattr(self, 'shuffle_channels', False)
        if roll_channels and shuffle_channels:
            raise ValueError("--roll-channels and --shuffle-channels are mutually exclusive")
        elif roll_channels:
            logger.info("-->Rolling channels transform<--")
            train_nww.append_transform(
                RollDimension(roll_dim=0, min_roll=0,
                              max_roll=train_nww.sensor_count - 1)
            )
        elif shuffle_channels:
            logger.info("-->Shuffle channels transform<--")
            train_nww.append_transform(
                ShuffleDimension()
            )

        if getattr(self, 'random_labels', False):
            logger.info("-->Randomizing target labels<--")
            train_nww.append_target_transform(
                RandomIntLike(low=0, high=2)
            )

        dataset_map['train'] = train_nww

        if cv_kws['patient_tuples'] is not None:
            print("+" * 50)
            print(f"Using {cv_kws['patient_tuples']}")
            dataset_map['cv'] = dataset_cls(sensor_columns=train_nww.selected_columns, **cv_kws)
        elif dataset_cls == HarvardSentences:
            logger.info("*" * 30)
            logger.info("Splitting on random key levels for harvard sentences (UCSD)")
            logger.info("*" * 30)
            _train, _cv = train_nww.split_select_random_key_levels()
            dataset_map.update(dict(train=_train, cv=_cv))
        else:
            print("~" * 100)
            from sklearn.model_selection import train_test_split
            train_ixs, cv_ixes = train_test_split(range(len(train_nww)))
            cv_nww = dataset_cls(data_from=train_nww, **cv_kws).select(cv_ixes)
            train_nww.select(train_ixs)
            dataset_map.update(dict(train=train_nww,
                                    cv=cv_nww))

        if test_kws['patient_tuples'] is not None:
            dataset_map['test'] = dataset_cls(  # power_q=options.power_q,
                sensor_columns=train_nww.selected_columns,
                **test_kws)
        else:
            logger.info(" - No test datasets provided - ")

        # dataset_map = dict(train=train_nww, cv=cv_nww, test=test_nww)
        if self.random_sensors_to_samples:
            additional_transforms = (list() if additional_transforms is None else additional_transforms)
            additional_transforms += [SelectFromDim(dim=0,
                                                    index='random',
                                                    keep_dim=True)]

        if isinstance(additional_transforms, list):
            dataset_map = {k: v.append_transform(additional_transforms)
                           for k, v in dataset_map.items()}

        dataloader_map = {k: v.to_dataloader(**dl_kws)
                          for k, v in dataset_map.items()}
        eval_dataloader_map = {k: v.to_dataloader(**eval_dl_kws)
                               for k, v in dataset_map.items()}

        return dataset_map, dataloader_map, eval_dataloader_map


@dataclass
class NorthwesternWordsDatasetOptions(DatasetOptions):
    dataset_name: str = 'nww'
    train_sets: str = 'MC-21-0'


@dataclass
class HarvardSentencesDatasetOptions(DatasetOptions):
    dataset_name: str = 'hvs'
    train_sets: str = 'UCSD-19'
    flatten_sensors_to_samples: bool = True
    extra_output_keys: Optional[str] = 'sensor_ras_coord_arr'


if __name__ == """__main__""":
    hvs_tuples = HarvardSentences.make_tuples_from_sets_str('UCSD-28')
    hvs = HarvardSentences(hvs_tuples, #flatten_sensors_to_samples=False,
                           extra_output_keys='sensor_ras_coord_arr',
                           pre_processing_pipeline='region_classification')
    print(hvs[0])

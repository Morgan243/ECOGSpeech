from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
import attr
import logging
from ecog_speech import utils

with_logger = utils.with_logger(prefix_name=__name__)


@attr.s
@with_logger
class DictTrf(BaseEstimator, TransformerMixin):
    def transform(self, data_map):
        in_keys = set(data_map.keys())
        updates = self.process(data_map)
        out_keys = set(updates.keys())
        self.logger.debug(f"Updated keys: {out_keys}")
        data_map.update(updates)
        return data_map

    def process(self, data_map):
        raise NotImplementedError()


@attr.s
@with_logger
class ParseTimeSeriesArrToFrame(DictTrf):
    array_key = attr.ib()
    fs_key = attr.ib()
    default_fs = attr.ib()
    dtype = attr.ib(None)
    reshape = attr.ib(None)
    output_key = attr.ib(None)

    def process(self, data_map):
        # determine what the output key will be
        arr_key = self.array_key if self.output_key is None else self.output_key

        try:
            self.logger.debug(f"Accessing {self.fs_key}")
            _fs = data_map[self.fs_key]
        except KeyError as ke:
            msg = f"provided fs_key={self.fs_key} not in data_map, expected one of: {list(data_map.keys())}"
            self.logger.warning(msg)
            _fs = self.default_fs

        self.logger.debug(f"Input source frequency, fs object: {_fs}")

        # Sometimes it's a scalar value inside some arrays
        if isinstance(_fs, np.ndarray):
            fs = data_map[self.fs_key].reshape(-1)[0]

        # Otherwise, just make sure it is integer
        fs = int(_fs)

        arr = data_map[self.array_key]

        if self.reshape is not None:
            arr = arr.reshape(self.reshape)

        ix = pd.TimedeltaIndex(pd.RangeIndex(0, arr.shape[0]) / fs, unit='s')
        if arr.ndim == 1:
            arr_df = pd.Series(arr, index=ix, dtype=self.dtype, name=arr_key)
        else:
            arr_df = pd.DataFrame(arr, index=ix, dtype=self.dtype)
        self.logger.info(f"{self.array_key}@{fs}, shape: {arr_df.shape}, [{arr_df.index[0], arr_df.index[-1]}]")

        return {self.fs_key: fs, arr_key: arr_df}


@attr.s
@with_logger
class IdentifyGoodAndBadSensors(DictTrf):
    electrode_qual_key = attr.ib('electrodes')
    on_missing = attr.ib('ignore')
    sensor_selection = attr.ib(None)
    good_electrode_ind_column = attr.ib(0)

    def process(self, data_map):
        k = self.electrode_qual_key
        if k not in data_map:
            msg = f"Electrodes with key {self.electrode_qual_key} not found among {list(data_map.keys())}"
            if self.on_missing == 'ignore':
                self.logger.warning(msg)
                return dict(good_sensor_columns=None, bad_sensor_columns=None)
            else:
                raise KeyError("ERROR: " + msg)

        chann_code_cols = ["code_%d" % e for e in range(data_map[k].shape[-1])]
        channel_df = pd.DataFrame(data_map['electrodes'], columns=chann_code_cols)
        self.logger.info("Found N electrodes = %d" % channel_df.shape[0])

        # required_sensor_columns = channel_df.index.tolist() if sensor_columns is None else sensor_columns
        # Mask for good sensors
        ch_m = (channel_df.iloc[:, self.good_electrode_ind_column] == 1)
        all_valid_sensors = ch_m[ch_m].index.tolist()

        # Spec the number of sensors that the ecog array mush have
        if self.sensor_selection is None:
            required_sensor_columns = channel_df.index.tolist()
        elif self.sensor_selection == 'valid':
            sensor_columns = all_valid_sensors
            required_sensor_columns = sensor_columns
        else:
            required_sensor_columns = self.sensor_selection

        #
        good_sensor_columns = [c for c in all_valid_sensors if c in required_sensor_columns]
        bad_sensor_columns = list(set(required_sensor_columns) - set(good_sensor_columns))
        return dict(good_sensor_columns=good_sensor_columns, bad_sensor_columns=bad_sensor_columns,
                    channel_status=channel_df)


@attr.s
@with_logger
class ApplySensorSelection(DictTrf):
    selection = attr.ib(None)
    signal_key = attr.ib('signal')
    bad_sensor_method = attr.ib('zero')

    def process(self, data_map):
        signal_df = data_map[self.signal_key]

        if self.selection is None:
            selected_cols = data_map.get('good_sensor_columns')
        elif isinstance(self.selection, list):
            selected_cols = self.selection
        else:
            raise ValueError(f"Don't understand selection: {self.selection}")

        if selected_cols is None:
            selected_cols = signal_df.columns.tolist()


        bs_cols = data_map['bad_sensor_columns']
        sel_signal_df = signal_df.copy()
        if bs_cols is not None and len(bs_cols) > 0:
            if self.bad_sensor_method == 'zero':
                signal_df.loc[:, bs_cols] = 0.
            elif self.bad_sensor_method == 'ignore':
                self.logger.warning(f"Ignoring bad sensor columns: {bs_cols}")
            else:
                raise KeyError(f"Don't understand bad_sensor_method: {self.bad_sensor_method}")

        return {self.signal_key: sel_signal_df}


@attr.s
@with_logger
class SubsampleSignal(DictTrf):
    signal_keys = attr.ib(('signal', 'stim'))
    signal_rate_key = attr.ib('fs_signal')
    rate = attr.ib(2)

    def process(self, data_map):
        output = {k: data_map[k].iloc[::self.rate] for k in self.signal_keys}
        output[self.signal_rate_key] = int((1. / self.rate) * data_map[self.signal_rate_key])
        return output


@attr.s
@with_logger
class PowerThreshold(DictTrf):
    stim_key = attr.ib('stim')
    audio_key = attr.ib('audio')

    speaking_threshold = attr.ib(0.005)
    silence_threshold = attr.ib(0.002)

    speaking_window_samples = attr.ib(48000)
    # More silence data, so require larger region of threshold check
    silence_window_samples = attr.ib(48000)
    stim_silence_value = attr.ib(0)
    silence_quantile_threshold = attr.ib(None)
    silence_n_smallest = attr.ib(None)
    speaking_quantile_threshold = attr.ib(None)

    def process(self, data_map):
        return self.power_threshold(data_map[self.audio_key], data_map[self.stim_key],
                                    speaking_threshold=self.speaking_threshold,
                                    speaking_window_samples=self.speaking_window_samples,
                                    silence_threshold=self.silence_threshold,
                                    silence_window_samples=self.silence_window_samples,
                                    stim_silence_value=self.stim_silence_value,
                                    speaking_quantile_threshold=self.speaking_quantile_threshold,
                                    silence_quantile_threshold=self.silence_quantile_threshold,
                                    silence_n_smallest=self.silence_n_smallest)

    @classmethod
    def power_threshold(cls, audio_s, stim_s, speaking_threshold,
                        speaking_window_samples,
                        silence_threshold,
                        silence_window_samples, stim_silence_value,
                        speaking_quantile_threshold,
                        silence_quantile_threshold,
                        silence_n_smallest):
        cls.logger.info("Power threshold")
        #### Speaking
        rolling_pwr = (audio_s
                       .abs().rolling(speaking_window_samples, center=True)
                       .median().reindex(stim_s.index, method='nearest').fillna(0))
                       #.max().reindex(stim_s.index, method='nearest').fillna(0))

        if speaking_quantile_threshold is not None:
            cls.logger.info("Using speaking quantile")
            speaking_quantile_threshold = float(speaking_quantile_threshold)
            thresholded_speaking_pwr = rolling_pwr.pipe(lambda s: s > s.quantile(speaking_quantile_threshold) )
        else:
            thresholded_speaking_pwr = (rolling_pwr > speaking_threshold)

        speaking_stim_auto_m = (stim_s != 0.) & thresholded_speaking_pwr

        #### Silence
        silence_rolling_pwr = (audio_s
                               .abs().rolling(silence_window_samples, center=True)
                               .median().reindex(stim_s.index, method='nearest').fillna(np.inf))
                               #.max().reindex(stim_s.index, method='nearest').fillna(0))

        if silence_n_smallest is not None:
            cls.logger.info(f"Using silence nsmallest on {type(silence_rolling_pwr)}")
            silence_n_smallest = int(silence_n_smallest)
            n_smallest = silence_rolling_pwr.nsmallest(silence_n_smallest)
            n_smallest_ix = n_smallest.index
            #cls.logger.info(f"n smallest: {n_smallest_ix.()}")
            thresholded_silence_pwr = pd.Series(False, index=silence_rolling_pwr.index)
            thresholded_silence_pwr.loc[n_smallest_ix] = True
        elif silence_quantile_threshold is not None:
            cls.logger.info("Using silence quantile")
            silence_quantile_threshold = float(silence_quantile_threshold)
            thresholded_silence_pwr = silence_rolling_pwr.pipe(lambda s: s < s.quantile(silence_quantile_threshold) )
        else:
            cls.logger.info("Using silence power-threshold")
            thresholded_silence_pwr = (silence_rolling_pwr < silence_threshold)

        #silence_stim_auto_m = (stim_s == stim_silence_value) & (~speaking_stim_auto_m) & thresholded_silence_pwr
        #silence_stim_auto_m = (stim_s == stim_silence_value) & thresholded_silence_pwr
        silence_stim_auto_m = (~speaking_stim_auto_m) & thresholded_silence_pwr

        # Is the number of unique word codes different when using the threshold selected subset we
        # just produced (stim_auto_m)?
        # - Subtract one for no speech (0)
        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[speaking_stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[speaking_stim_auto_m].nunique(False))
            #print(msg)
            cls.logger.warning(msg)

        # Create a new stim array with original word code where it's set, otherwise zero
        stim_pwrt_s = pd.Series(np.where(speaking_stim_auto_m, stim_s, 0), index=stim_s.index)
        stim_pwrt_diff_s = stim_pwrt_s.diff().fillna(0).astype(int)

        silence_stim_pwrt_s = pd.Series(np.where(silence_stim_auto_m, 1, 0), index=stim_s.index)
        silence_stim_pwrt_diff_s = silence_stim_pwrt_s.diff().fillna(0).astype(int)

        #coded_silence_stim = (silence_stim_pwrt_diff_s.cumsum() + 1) * silence_stim_pwrt_s
        coded_silence_stim = (silence_stim_pwrt_s.diff().eq(-1).cumsum() + 1) * silence_stim_pwrt_s

        updates = dict(stim_pwrt=stim_pwrt_s, stim_pwrt_diff=stim_pwrt_diff_s,
                       silence_stim_pwrt_s=silence_stim_pwrt_s, silence_stim_pwrt_diff_s=silence_stim_pwrt_diff_s,
                       coded_silence_stim=coded_silence_stim,
                       rolling_audio_pwr=rolling_pwr)
        return updates


@attr.s
@with_logger
class StimFromStartStopWordTimes(DictTrf):
    stim_speaking_value = attr.ib(51)

    def process(self, data_map):
        word_df = pd.DataFrame(data_map['start_stop_word_ms'],
                               columns=['start_t', 'stop_t', 'word'])

        stim = data_map['stim']

        speaking_stim_s = stim[stim.lt(self.stim_speaking_value) & stim.gt(0)]

        # convert to time in secodes
        word_df['start_t'] = word_df.start_t.astype(float).apply(lambda v: pd.Timedelta(v, 's'))
        word_df['stop_t'] = word_df.stop_t.astype(float).apply(lambda v: pd.Timedelta(v, 's'))

        # Get the  index nearest to the words start time from the listening index
        sent_code_ixes = word_df.start_t.apply(lambda v: speaking_stim_s.index.get_loc(v, method='nearest'))
        # Get the nearest to the words start time for the stim values
        start_ixes = word_df.start_t.apply(lambda v: stim.index.get_loc(v, method='nearest'))

        word_df['stim_start_t'] = stim.index[start_ixes]

        word_df['stim_sentcode'] = speaking_stim_s.iloc[sent_code_ixes].values
        word_df['stim_sentcode_t'] = stim.index[sent_code_ixes]

        word_df = word_df.set_index('stim_start_t').join(stim)

        # Extract sentence level stim
        sent_df = pd.concat([word_df.groupby('stim_sentcode').start_t.min(),
                             word_df.groupby('stim_sentcode').stop_t.max()], axis=1)

        sent_df['length'] = sent_df.diff(axis=1).stop_t.rename('length')
        # sent_df = sent_df.join(sent_df.diff(axis=1).stop_t.rename('length'))

        ix = stim.index
        sentence_stim = pd.Series(0, index=ix)
        word_stim = pd.Series(0, index=ix)

        # # #
        for i, (gname, gdf) in enumerate(word_df.groupby('stim_sentcode')):
            start_t = gdf.start_t.min()
            stop_t = gdf.stop_t.max()

            start_i = sentence_stim.index.get_loc(start_t, method='nearest')
            stop_i = sentence_stim.index.get_loc(stop_t, method='nearest')

            sentence_stim.iloc[start_i: stop_i] = sentence_stim.max() + 1

            is_word_m = gdf.word.str.upper() == gdf.word

            for ii, (_gname, _gdf) in enumerate(gdf[is_word_m].groupby('word')):
                _start_t = _gdf.start_t.min()
                _stop_t = _gdf.stop_t.max()

                _start_i = sentence_stim.index.get_loc(_start_t, method='nearest')
                _stop_i = sentence_stim.index.get_loc(_stop_t, method='nearest')
                word_stim.iloc[_start_i: _stop_i] = word_stim.max() + 1

        return dict(word_start_stop_times=word_df,
                    sent_start_stop_time=sent_df,
                    word_stim=word_stim, sentence_stim=sentence_stim)


def object_as_key_or_itself(key_or_value, remap=None):
    if isinstance(remap, dict):
        value = remap[key_or_value]
    elif remap is not None:
        value = remap
    elif remap is None:
        value = key_or_value
    else:
        raise ValueError(f"Dont know how to handle remap of type: {type(remap )}")
    return value


@attr.s
@with_logger
class WindowSampleIndicesFromIndex(DictTrf):
    stim_key = attr.ib('stim')
    fs_key = attr.ib('fs_signal')
    index_shift = attr.ib(None)
    stim_target_value = attr.ib(1)
    window_size = attr.ib(pd.Timedelta(0.5, 's'))
    stim_value_remap = attr.ib(None)

    def process(self, data_map):
        return self.make_sample_indices(data_map[self.stim_key], data_map[self.fs_key], win_size=self.window_size,
                                        index_shift=self.index_shift,
                                        stim_target_value=self.stim_target_value, stim_value_remap=self.stim_value_remap,
                                        existing_sample_indices_map=data_map.get('sample_index_map'),)

    @classmethod
    def make_sample_indices(cls, stim, fs, win_size,
                            index_shift, stim_target_value, stim_value_remap,
                            existing_sample_indices_map):
        index_shift = pd.Timedelta(0, 's') if index_shift is None else index_shift
        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        sample_indices = dict()
        expected_window_samples = int(fs * win_size.total_seconds())

        target_indexes = (stim == stim_target_value).pipe(lambda s: s[s].index.tolist())
        target_indices = [stim.loc[offs + index_shift:offs + win_size + index_shift].iloc[:expected_window_samples].index
                          for offs in target_indexes
                          if len(stim.loc[offs + index_shift:offs + win_size + index_shift]) >= expected_window_samples]

#        if isinstance(stim_value_remap, dict):
#            stim_key = stim_value_remap[stim_target_value]
#        elif stim_value_remap is not None:
#            stim_key = stim_value_remap
#        elif stim_value_remap is None:
#            stim_key = stim_target_value
#        else:
#            raise ValueError(f"Dont know how to handle stim_value_remap of type: {type(stim_value_remap)}")
        stim_key = object_as_key_or_itself(stim_target_value, stim_value_remap)
        sample_indices[stim_key] = sample_indices.get(stim_key, list()) + target_indices

        if existing_sample_indices_map is not None:
            existing_sample_indices_map.update(sample_indices)
            sample_indices = existing_sample_indices_map
        return dict(sample_index_map=sample_indices, n_samples_per_window=expected_window_samples)


@attr.s
@with_logger
class WindowSampleIndicesFromStim(DictTrf):

    stim_key = attr.ib('stim')
    fs_key = attr.ib('fs_signal')
    window_size = attr.ib(pd.Timedelta(0.5, 's'))

    # One of rising or falling
    target_onset_reference = attr.ib('rising')
    target_offset_reference = attr.ib('falling')
    target_onset_shift = attr.ib(pd.Timedelta(-0.50, 's'))
    target_offset_shift = attr.ib(pd.Timedelta(0., 's'))

    max_target_region_size = attr.ib(600)
    stim_value_remap = attr.ib(None)

    def process(self, data_map):
        return self.make_sample_indices(data_map[self.stim_key], data_map[self.fs_key],
                                        win_size=self.window_size,
                                        target_onset_ref=self.target_onset_reference,
                                        target_onset_shift=self.target_onset_shift,
                                        target_offset_ref=self.target_offset_reference,
                                        target_offset_shift=self.target_offset_shift,
                                        #silence_value=self.silence_stim_value, silence_samples=self.silence_samples,
                                        #silent_window_scale=self.silent_window_scale,
                                        max_target_region_size=self.max_target_region_size,
                                        existing_sample_indices_map=data_map.get('sample_index_map'),
                                        stim_value_remap=self.stim_value_remap)

    @classmethod
    def make_sample_indices(cls, stim, fs, win_size, target_onset_ref, target_onset_shift,
                            target_offset_ref, target_offset_shift,
                            #silence_value, silence_samples, silent_window_scale,
                            max_target_region_size, existing_sample_indices_map=None, stim_value_remap=None):

        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        expected_window_samples = int(fs * win_size.total_seconds())
        #label_region_sample_size = int(fs * label_region_size.total_seconds())
        cls.logger.info((fs, win_size))
        cls.logger.info("Samples per window: %d" % expected_window_samples)

        # Will map of codes to list of indices into the stim signal:
        # word_code->List[pd.Index, pd.Index, ...]
        sample_indices = dict()

        # TODO: This will not work for constant stim value (i.e. True/False, 1/0)?
        # TODO: Need to review UCSD data and how to write something that will work for its regions
        s_grp = stim[stim > 0].pipe(lambda _s: _s.groupby(_s))
        for stim_value, g_s in tqdm(s_grp, desc=f"Processing stim regions"):
            start_t = g_s.index.min()
            stop_t = g_s.index.max()

            if target_onset_ref == 'rising':
                target_start_t = start_t + target_onset_shift
            elif target_onset_ref == 'falling':
                target_start_t = stop_t + target_onset_shift
            else:
                raise ValueError(f"Dont understand {target_onset_ref}")

            if target_offset_ref == 'rising':
                target_stop_t = start_t + target_offset_shift
            elif target_offset_ref == 'falling':
                target_stop_t = stop_t + target_offset_shift
            else:
                raise ValueError(f"Dont understand {target_offset_ref}")

            # Get the window starting indices for each region of interest
            # Note on :-expected_window_samples
            #   - this removes the last windows worth since windows starting here would have out of label samples
            # Commented this out - use the offsets to handle this?
            #target_start_ixes = stim[target_start_t:target_stop_t].index.tolist()#[:-expected_window_samples]
            target_start_ixes = stim[target_start_t:target_stop_t].index.tolist()#[:-expected_window_samples]

            # Go through the labeled region indices and pull a window of data
            target_indices = [stim.loc[offs:offs + win_size].iloc[:expected_window_samples].index
                                for offs in target_start_ixes[:max_target_region_size]
                                    if len(stim.loc[offs:offs + win_size]) >= expected_window_samples]

#            if isinstance(stim_value_remap, dict):
#                stim_key = stim_value_remap[stim_value]
#            elif stim_value_remap is not None:
#                stim_key = stim_value_remap
#            elif stim_value_remap is None:
#                stim_key = stim_value
#            else:
#                raise ValueError(f"Dont know how to handle stim_value_remap of type: {type(stim_value_remap)}")

            stim_key = object_as_key_or_itself(stim_value, stim_value_remap)
            sample_indices[stim_key] = sample_indices.get(stim_key, list()) + target_indices

        # Go through all samples - make noise if sample size is off (or should throw error?)
        for k, _s in sample_indices.items():
            for i, _ixs in enumerate(_s):
                if len(_ixs) != expected_window_samples:
                    cls.logger.warning(f"[{k}][{i}] ({len(_ixs)}): {_ixs}")

        # Debug code printing the unique lengths of each window for each word code
        #print({k : sorted(list(set(map(len, _s)))) for k, _s in sample_indices.items()})
        if existing_sample_indices_map is not None:
            existing_sample_indices_map.update(sample_indices)
            sample_indices = existing_sample_indices_map
        return dict(sample_index_map=sample_indices, n_samples_per_window=expected_window_samples)


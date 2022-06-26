from sklearn.base import BaseEstimator, TransformerMixin
import torchaudio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
import attr
import logging
from ecog_speech import utils
import torch

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
        assert arr_df.index.is_unique, f"NON UNIQUE TIME SERIES INDEX FOR KEY {self.array_key}"

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
            msg = f"Electrodes with key '{self.electrode_qual_key}' not found among {list(data_map.keys())}"
            if self.on_missing == 'ignore':
                self.logger.warning(msg + ' - but on_missing="ignore", so moving on')
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

    @classmethod
    def select_from_ras(cls, data_map, selected_columns, bad_columns, **kws):
        s_df = data_map['sensor_ras_df'].loc[selected_columns]
        return {'sensor_ras_df': s_df,
                'sensor_ras_coord_arr': s_df.filter(like='coord').values
                }

    def process(self, data_map):
        signal_df = data_map[self.signal_key]

        if self.selection is None:
            self.logger.info("Checking data_map for good_sensor_columns")
            selected_cols = data_map.get('good_sensor_columns')
            self.logger.inf(f"Got: {selected_cols}")
        elif isinstance(self.selection, list):
            self.logger.info(f"Selection of columns passed to sensor selection")
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

        r_val = {self.signal_key: sel_signal_df,
                 'selected_columns': selected_cols,
                 'bad_columns': bs_cols,
                 'bad_sensor_method': self.bad_sensor_method}

        # TODO: a way to apply a bunch of selection functions
        if 'sensor_ras_df' in data_map:
            self.logger.info("Selecting columns in RAS coordinate data")
            ras_sel = self.select_from_ras(data_map, **r_val)
            r_val.update(ras_sel)

        return r_val


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
class StandardNormSignal(DictTrf):
    signal_key = attr.ib('signal')
    output_key = attr.ib('signal')
    rate = attr.ib(2)

    def process(self, data_map):
        df = data_map[self.signal_key]
        mu = df.mean()
        std = df.std()
        return {self.output_key: (df - mu) / std}


@attr.s
@with_logger
class ExtractMFCC(DictTrf):

    n_fft = attr.ib(1024)
    win_length = attr.ib(None)
    hop_length = attr.ib(512)
    n_mels = attr.ib(13)
    fs = attr.ib(None)

    audio_key = attr.ib('audio')
    audio_fs_key = attr.ib('fs_audio')

    def process(self, data_map):
        audio_s = data_map[self.audio_key]
        fs = data_map[self.audio_fs_key]

        if not hasattr(self, 'melspec_trf'):
            self.melspec_trf = torchaudio.transforms.MelSpectrogram(fs,
                                                           n_fft=self.n_fft,
                                                           win_length=self.win_length,
                                                           hop_length=self.hop_length,
                                                           center=True,
                                                           pad_mode="reflect",
                                                           power=2.0,
                                                           norm="slaney",
                                                           onesided=True,
                                                           n_mels=self.n_mels,
                                                           mel_scale="htk")

        audio_arr = torch.from_numpy(audio_s.values).float()
        audio_mfc = self.melspec_trf(audio_arr)

        audio_mfc_df = pd.DataFrame(audio_mfc.T.detach().cpu())
        mfc_ix = pd.TimedeltaIndex(pd.RangeIndex(0, audio_mfc_df.shape[0]) / (fs / self.melspec_trf.hop_length),
                                   unit='s')
        audio_mfc_df.index = mfc_ix

        return dict(audio_mel_spec=audio_mfc_df)


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
    n_silence_windows = attr.ib(35000)
    speaking_quantile_threshold = attr.ib(None)

    def process(self, data_map):
        return self.power_threshold(data_map[self.audio_key], data_map[self.stim_key],
                                    speaking_threshold=self.speaking_threshold,
                                    speaking_window_samples=self.speaking_window_samples,
                                    silence_threshold=self.silence_threshold,
                                    silence_window_samples=self.silence_window_samples,
                                    stim_silence_value=self.stim_silence_value,
                                    n_silence_windows=self.n_silence_windows,
                                    speaking_quantile_threshold=self.speaking_quantile_threshold,
                                    silence_quantile_threshold=self.silence_quantile_threshold,
                                    silence_n_smallest=self.silence_n_smallest)

    @classmethod
    def power_threshold(cls, audio_s, stim_s, speaking_threshold,
                        speaking_window_samples,
                        silence_threshold,
                        silence_window_samples, stim_silence_value,
                        n_silence_windows,
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
            cls.logger.info(f"Using speaking quantile {speaking_quantile_threshold}")
            speaking_quantile_threshold = float(speaking_quantile_threshold)
            thresholded_speaking_pwr = rolling_pwr.pipe(lambda s: s > s.quantile(speaking_quantile_threshold))
        else:
            thresholded_speaking_pwr = (rolling_pwr > speaking_threshold)

        speaking_stim_auto_m = (stim_s != stim_silence_value) & thresholded_speaking_pwr

        #### Silence
        silence_rolling_pwr = (audio_s
                               .abs().rolling(silence_window_samples, center=True)
                               .median().reindex(stim_s.index, method='nearest').fillna(np.inf))
                               #.max().reindex(stim_s.index, method='nearest').fillna(0))

        if silence_n_smallest is not None:
            silence_n_smallest = int(silence_n_smallest)
            cls.logger.info(f"Using silence {silence_n_smallest} smallest on {type(silence_rolling_pwr)}")
            n_smallest = silence_rolling_pwr.nsmallest(silence_n_smallest)
            n_smallest_ix = n_smallest.index
            #cls.logger.info(f"n smallest: {n_smallest_ix.()}")
            thresholded_silence_pwr = pd.Series(False, index=silence_rolling_pwr.index)
            thresholded_silence_pwr.loc[n_smallest_ix] = True
        elif silence_quantile_threshold is not None:
            cls.logger.info("Using silence quantile")
            silence_quantile_threshold = float(silence_quantile_threshold)
            thresholded_silence_pwr = silence_rolling_pwr.pipe(lambda s: s <= s.quantile(silence_quantile_threshold) )
        else:
            cls.logger.info("Using silence power-threshold")
            thresholded_silence_pwr = (silence_rolling_pwr < silence_threshold)

        #silence_stim_auto_m = (stim_s == stim_silence_value) & (~speaking_stim_auto_m) & thresholded_silence_pwr
        #silence_stim_auto_m = (stim_s == stim_silence_value) & thresholded_silence_pwr
        silence_stim_auto_m = (~speaking_stim_auto_m) & thresholded_silence_pwr

        if n_silence_windows is not None and n_silence_windows > 0:
            available_silence_stim: float = silence_stim_auto_m.sum()
            cls.logger.info(f"Sampling {n_silence_windows} from {available_silence_stim}")
            kws = dict(replace=False)
            if n_silence_windows > available_silence_stim:
                cls.logger.warning("More silent stims requested than available (see above INFO) - sampling with replace")
                kws['replace'] = True
            silence_samples = silence_stim_auto_m[silence_stim_auto_m].sample(n_silence_windows, **kws)
            silence_stim_auto_m = pd.Series(False, index=silence_stim_auto_m.index)
            silence_stim_auto_m[silence_samples.index] = True

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
class SentCodeFromStartStopWordTimes(DictTrf):
    """Listening code during the listening region"""
    stim_speaking_value = attr.ib(51)

    @classmethod
    def parse_start_stop_word_ms(cls, sswms):
        word_df = pd.DataFrame(sswms,
                               columns=['start_t', 'stop_t', 'word'])
        # convert to time in secodes
        word_df['start_t'] = word_df.start_t.astype(float).apply(lambda v: pd.Timedelta(v, 's'))
        word_df['stop_t'] = word_df.stop_t.astype(float).apply(lambda v: pd.Timedelta(v, 's'))

        return word_df

    def process(self, data_map):
        word_df = self.parse_start_stop_word_ms(data_map['start_stop_word_ms'])
        # Stim:
        stim = data_map['stim']

        # speaking is lowest stim code - find all word codes for when they are listening (lt(stim_speaking))
        listening_stim_s = stim[stim.lt(self.stim_speaking_value) & stim.gt(0)]

        # Get the listening sample nearest to the words start time from the listening index
        start_listening_ixes = listening_stim_s.index.get_indexer(word_df.start_t.values, method='nearest')
        # Get the index nearest to the words start time for the stim values - should basically be the start_t value
        start_stim_ixes = stim.index.get_indexer(word_df.start_t, method='nearest')

        # This should essentially match the start_t column
        word_df['stim_start_t'] = stim.index[start_stim_ixes]

        # Get the stim code (sentence code) neared to the at the point closest to each rows start_t
        word_df['stim_sentcode'] = listening_stim_s.iloc[start_listening_ixes].values
        # Get the time of that stim code (sentence code) neared to the at the point closest to each rows start_t
        word_df['stim_sentcode_t'] = listening_stim_s.iloc[start_listening_ixes].index

        # add the stim in just cause - it should always be 51 (?) for speaking code since this is forced alignment data
        word_df = word_df.set_index('stim_start_t').join(stim)

        ### ----
        # NOTE:Check for repeated stim codes (..only seen in UCSD 28, sent code 45), this adds sent code 45.5 to their stim
        grps, max_delta =  list(), pd.Timedelta(1, 'm')
        for sent_code, sc_df in word_df.groupby('stim_sentcode'):
            delta_t = sc_df.stim_sentcode_t.max() - sc_df.stim_sentcode_t.min()
            o_cs_df = sc_df

            if delta_t > max_delta:
                self.logger.warning(f"Sent code {sent_code} has a time range more than a minute: {delta_t}")
                # Limit to the onset markers and find the word start (speaking
                # start) with the biggest jump in sent code time
                split_point_t = sc_df[sc_df.word.eq('on')].stim_sentcode_t.diff().idxmax()
                # Grab the first instance, dropping the last sample that is actually from the second instance
                first_w_df = sc_df.loc[:split_point_t].iloc[:-1].copy()
                # Get the last stim sent code for the first instance of the duplicate sent code
                split_point_t = first_w_df.iloc[-1].stop_t

                # The latter portion of the word df, after the split
                last_w_df = sc_df.loc[split_point_t:].copy()
                # Give it a word code that does't exist, but clear where it came from , so + 0.5
                last_w_df['stim_sentcode'] = last_w_df['stim_sentcode'] + 0.5

                # WARNING: Change the stim - replace instances past the split point with the new code
                stim.loc[split_point_t:] = stim.loc[split_point_t:].replace(sent_code, sent_code + 0.5)

                # put it all together
                o_cs_df = pd.concat([
                    first_w_df,
                    last_w_df
                ])

            grps.append(o_cs_df)

        word_df = pd.concat(grps).sort_index()
        # END
        ### ----

        # Extract sentence level stim
        sent_df = pd.concat([word_df.groupby('stim_sentcode').start_t.min(),
                             word_df.groupby('stim_sentcode').stop_t.max()], axis=1)

        sent_df['length'] = sent_df.diff(axis=1).stop_t.rename('length')
        # sent_df = sent_df.join(sent_df.diff(axis=1).stop_t.rename('length'))

        #ix = stim.index
        #sentence_stim = pd.Series(0, index=ix)
        #word_stim = pd.Series(0, index=ix)

        ## # #
        #for i, (gname, gdf) in enumerate(word_df.groupby('stim_sentcode')):
        #    start_t = gdf.start_t.min()
        #    stop_t = gdf.stop_t.max()

        #    start_i = sentence_stim.index.get_loc(start_t, method='nearest')
        #    stop_i = sentence_stim.index.get_loc(stop_t, method='nearest')

        #    # Set this sentence to some incrementing indicator
        #    sentence_stim.iloc[start_i: stop_i] = sentence_stim.max() + 1

        #    # Spoken is word is all caps string
        #    is_word_m = gdf.word.str.upper() == gdf.word

        #    # Set each words region in this sentence within the word_stim
        #    for ii, (_gname, _gdf) in enumerate(gdf[is_word_m].groupby('word')):
        #        _start_t = _gdf.start_t.min()
        #        _stop_t = _gdf.stop_t.max()

        #        _start_i = sentence_stim.index.get_loc(_start_t, method='nearest')
        #        _stop_i = sentence_stim.index.get_loc(_stop_t, method='nearest')
        #        word_stim.iloc[_start_i: _stop_i] = word_stim.max() + 1

        return dict(word_start_stop_times=word_df,
                    sent_start_stop_time=sent_df,
                    stim=stim
                    #word_stim=word_stim, sentence_stim=sentence_stim
                    )

@with_logger
@attr.s
class NewNewMultiTaskStartStop(DictTrf):
    def process(self, data_map):
        wsst_df = data_map['word_start_stop_times'].copy()
        stim = data_map['stim'].copy()

        wsst_df['speaking_word_length_t'] = wsst_df.stop_t - wsst_df.start_t

        sent_code_reffs_d = {sent_code: sent_df.start_t - sent_df.start_t.min()
                             for sent_code, sent_df in wsst_df.groupby('stim_sentcode')}

        offset_from_start_s = pd.concat(sent_code_reffs_d.values(), axis=0).rename('time_from_speaking_start')
        wsst_df = wsst_df.join(offset_from_start_s)


        listening_sent_stim_s = stim[stim.between(1, 50)]
        trial_regions_l = list()
        for sent_code, sent_s in listening_sent_stim_s.groupby(listening_sent_stim_s):
            trial_start_t = sent_s.index.min()
            future_listening_s = listening_sent_stim_s.loc[sent_s.index.max():]
            future_ne_s = future_listening_s[future_listening_s.ne(sent_code)]
            if len(future_ne_s) > 0:
                start_of_next_t = future_ne_s.index.min()
            else:
                self.logger.warning(f"Sent code {sent_code} has no future sentence dodes in front of it")
                future_s = stim.loc[sent_s.index.max():]
                start_of_next_t = future_s.index.max()

            #trial_end_t = listening_sent_stim_s.loc[:start_of_next_t].index[-1]
            trial_end_t = stim.loc[:start_of_next_t].index[-1]
            trial_regions_l.append(dict(stim_sentcode=sent_code, trial_start_t=trial_start_t, trial_stop_t=trial_end_t))

        trial_regions_df = pd.DataFrame(trial_regions_l)
        regions_l = list()

        def _work(s):
            return {f'{s.name}_start_t': s.index.min(),
                    f'{s.name}_stop_t': s.index.max()}

        for ix, r in trial_regions_df.iterrows():
            _s = stim.loc[r.trial_start_t: r.trial_stop_t]
            _s = _s[_s.isin((r.stim_sentcode, 0, 51, 52, 53))]
            regions_l.append(dict(
                stim_sentcode=r.stim_sentcode,
                **_s[_s.eq(r.stim_sentcode)].rename('listening_region').pipe(_work),
                **_s[_s.eq(51)].rename('speaking_region').pipe(_work),
                **_s[_s.eq(52)].rename('imagining_region').pipe(_work),
                **_s[_s.eq(53)].rename('mouthing_region').pipe(_work),
            ))

        regions_df = pd.DataFrame(regions_l)

        wsst_df = wsst_df.merge(trial_regions_df, on='stim_sentcode').merge(regions_df, on='stim_sentcode')  #

        start_offs_s = wsst_df.time_from_speaking_start
        stop_offs_s = wsst_df.time_from_speaking_start + wsst_df.speaking_word_length_t
        wsst_df = wsst_df.assign(
            listening_word_start_t=wsst_df.listening_region_start_t + start_offs_s,
            listening_word_stop_t=wsst_df.listening_region_start_t + stop_offs_s,

            speaking_word_start_t=wsst_df.speaking_region_start_t + start_offs_s,
            speaking_word_stop_t=wsst_df.speaking_region_start_t + stop_offs_s,

            mouthing_word_start_t=wsst_df.mouthing_region_start_t + start_offs_s,
            mouthing_word_stop_t=wsst_df.mouthing_region_start_t + stop_offs_s,

            imagining_word_start_t=wsst_df.imagining_region_start_t + start_offs_s,
            imagining_word_stop_t=wsst_df.imagining_region_start_t + stop_offs_s,
        )

        wsst_df = wsst_df.set_index('start_t', drop=False)
        wsst_df.index.name = 'stim_start_t'

        return dict(word_start_stop_times=wsst_df)
        
        
@attr.s
@with_logger
class NewMultiTaskStartStop(DictTrf):

    @classmethod
    def start_t_of_value(cls, s: pd.Series, name=None):
        name = f"{s.name}_start_t" if name is None else name

        r_s = (
            s.pipe(
            # TODO: maybe first off of groupby would work, but not sure it sorts index first...
            lambda s: s.groupby(s).apply(lambda s: s.sort_index().head(1)).reset_index(0, drop=True))
                                  .pipe(lambda s: pd.Series(s.index, index=s.values, name=name))#name='listening_region_start_t'))
                                  )
        return r_s

    @classmethod
    def stop_t_of_value(cls, s: pd.Series, name=None):
        name = f"{s.name}_stop_t" if name is None else name

        r_s = (
            s.pipe(
            # TODO: maybe first off of groupby would work, but not sure it sorts index first...
            lambda s: s.groupby(s).apply(lambda s: s.sort_index().tail(1)).reset_index(0, drop=True))
                                  .pipe(lambda s: pd.Series(s.index, index=s.values, name=name))#name='listening_region_start_t'))
                                  )
        return r_s

    @classmethod
    def region_start_stop_from_masks(cls, s, index_name='stim_sentcode', **masks):
        o_d = dict()
        for sname, s_mask in masks.items():
            _s = s[s_mask]
            start_s = cls.start_t_of_value(_s, name=sname + '_start_t')
            stop_s = cls.stop_t_of_value(_s, name=sname + '_stop_t')

            o_d[sname] = pd.concat([start_s, stop_s], axis=1)
            o_d[sname].index = o_d[sname].index.rename(index_name)

        return o_d

    def process(self, data_map):
        _word_df = data_map['word_start_stop_times'].copy()
        stim = data_map['stim']

        _word_df['speaking_length_t'] = _word_df.stop_t - _word_df.start_t

        sent_code_reffs_d = {sent_code: sent_df.start_t - sent_df.start_t.min()
                             for sent_code, sent_df in _word_df.groupby('stim_sentcode')}

        offset_from_start_s = pd.concat(sent_code_reffs_d.values(), axis=0).rename('time_from_speaking_start')
        _word_df = _word_df.join(offset_from_start_s)

        mask_d = {
            'listening_region': stim.lt(51) & stim.gt(0),

            #'speaking_region': stim.eq(51),
            #'imagining_region': stim.eq(52),
            #'mouthing_region': stim.eq(53),
        }


        stim_start_stop_d = self.region_start_stop_from_masks(stim, **mask_d)


#        _d = self.region_start_stop_from_masks(_word_df['stim_sentcode'],
#        **{
#            'speaking_region': _word_df.eq(51),
#            'imagining_region': stim.eq(52),
#            'mouthing_region': stim.eq(53),
#        }
#                                               )

        word_m_df = _word_df.merge(
            pd.concat(stim_start_stop_d.values(), axis=1),
            left_on='stim_sentcode', right_index=True)


        #regions_d = {
        #            'speaking_region': stim.eq(51),
        #            'imagining_region': stim.eq(52),
        #            'mouthing_region': stim.eq(53),
        #        }
        #for region_name, region_mask in regions_d.items():

        ######
        # For every speaking stim (i.e. =51) location, find the locations nearest the listening stop time
        _is = stim[stim.eq(51)].index.get_indexer(word_m_df['listening_region_stop_t'].values,
                                                  method='nearest')
        # The times in the stim are where speaking started
        word_m_df['speaking_region_start_t'] = stim[stim.eq(51)].iloc[_is].index
        #__is = stim[stim.eq(51)].index.get_indexer(stim[stim.eq(53)].index, method='nearest')
        #word_m_df['speaking_region_stop_t'] = stim[stim.eq(51)].iloc[__is].index

        ######
        # Then mouthing is performed so reference speaking
        _is = stim[stim.eq(53)].index.get_indexer(word_m_df['speaking_region_start_t'].values,
                                                  method='nearest')
        word_m_df['mouthing_region_start_t'] = stim[stim.eq(53)].iloc[_is].index
        #__is = stim[stim.eq(53)].index.get_indexer(stim[stim.eq(52)].index, method='nearest')
        #word_m_df['mouthing_region_stop_t'] = stim[stim.eq(53)].iloc[__is].index

        # Imagine is performed last, reference the mouthing start before it
        _is = stim[stim.eq(52)].index.get_indexer(word_m_df['mouthing_region_start_t'].values,
                                                  method='nearest')
        word_m_df['imagining_region_start_t'] = stim[stim.eq(52)].iloc[_is].index
        #__is = stim[stim.eq(52)].index.get_indexer(stim[stim.lt(51)].index, method='nearest')
        #word_m_df['imagining_region_stop_t'] = stim[stim.eq(52)].iloc[__is].index

        start_offs_s = word_m_df.time_from_speaking_start
        stop_offs_s = word_m_df.time_from_speaking_start + word_m_df.speaking_length_t
        word_m_df = word_m_df.assign(
            listening_word_start_t=word_m_df.listening_region_start_t + start_offs_s,
            listening_word_stop_t=word_m_df.listening_region_start_t + stop_offs_s,

            speaking_word_start_t=word_m_df.speaking_region_start_t + start_offs_s,
            speaking_word_stop_t=word_m_df.speaking_region_start_t + stop_offs_s,

            mouthing_word_start_t=word_m_df.mouthing_region_start_t + start_offs_s,
            mouthing_word_stop_t=word_m_df.mouthing_region_start_t + stop_offs_s,

            imagining_word_start_t=word_m_df.imagining_region_start_t + start_offs_s,
            imagining_word_stop_t=word_m_df.imagining_region_start_t + stop_offs_s,
        )
        #word_m_df['listening_word_start_t'] = word_m_df.listening_region_start_t + word_m_df.time_from_speaking_start

        #word_m_df.listening_region_start_t + word_m_df.time_from_speaking_start + word_m_df.speaking_length_t

        #for region_name in mask_d.keys():
        #    start_col, stop_col = region_name + '_start_t', region_name + '_stop_t'
        #    word_start_col, word_stop_col = start_col.replace('region', 'word'), stop_col.replace('region', 'word')
        #    word_m_df[word_start_col] = word_m_df[start_col] + offset_from_start_s
        #    word_m_df[word_stop_col] = word_m_df[start_col] + offset_from_start_s + word_m_df['speaking_length_t']

        return dict(word_start_stop_times=word_m_df)


#        self.logger.info("Hard coded to: Less than 51 and greater than zero")
#        s = stim[stim.lt(51) & stim.gt(0)]
#        start_listening_region = self.start_t_of_value(s, name='listening_region_start_t')
#        start_listening_region.index = start_listening_region.index.rename('stim_sentcode')
#
#        end_listening_region = self.stop_t_of_value(s, name='listening_region_stop_t')
#        end_listening_region.index = end_listening_region.index.rename('stim_sentcode')
#
#        #        start_listening_region = (stim[stim.lt(51) & stim.gt(0)]
##                                  .pipe(
##            # TODO: maybe first off of groupby would work, but not sure it sorts index first...
##            lambda s: s.groupby(s).apply(lambda s: s.sort_index().head(1)).reset_index(0, drop=True))
##                                  .pipe(lambda s: pd.Series(s.index, index=s.values, name='listening_region_start_t'))
##                                  )
##
##        start_listening_region.index = start_listening_region.index.rename('stim_sentcode')
##        end_listening_region = (stim[stim.lt(51) & stim.gt(0)]
##                                .pipe(
##            # TODO: maybe first off of groupby would work, but not sure it sorts index first...
##            lambda s: s.groupby(s).apply(lambda s: s.sort_index().tail(1)).reset_index(0, drop=True))
##                                .pipe(lambda s: pd.Series(s.index, index=s.values, name='listening_region_stop_t'))
##                                )
##
##        end_listening_region.index = end_listening_region.index.rename('stim_sentcode')
#
#        word_m_df = _word_df.merge(pd.concat([start_listening_region, end_listening_region], axis=1),
#                                    left_on='stim_sentcode', right_index=True)


# Create multi-task start stop that extracts the start and stop times
# Create general stim from start stop times
# Separate out current implementation of stim from startstop stimes to be sentence start stop extract?
@attr.s
@with_logger
class MultiTaskStartStop(DictTrf):
    """
     extract the same start stop times, only shifted to align in offset from the start of stims in the parameter value
    map
    """
    stim_val_map = attr.ib()

    @stim_val_map.default
    def stim_val_map_factory(self):
        return ({
            52: 'imagine',
            53: 'mouth',
        })

    def process(self, data_map):
        _word_df = data_map['word_start_stop_times'].copy()
        stim = data_map['stim']

        self.logger.info("Hard coded to: Less than 51 and greater than zero")
        start_listening_region = (stim[stim.lt(51) & stim.gt(0)]
                                  .pipe(
            # TODO: maybe first off of groupby would work, but not sure it sorts index first...
            lambda s: s.groupby(s).apply(lambda s: s.sort_index().head(1)).reset_index(0, drop=True))
                                  .pipe(lambda s: pd.Series(s.index, index=s.values, name='listening_region_start_t'))
                                  )

        start_listening_region.index = start_listening_region.index.rename('stim_sentcode')
        end_listening_region = (stim[stim.lt(51) & stim.gt(0)]
                                .pipe(
            # TODO: maybe first off of groupby would work, but not sure it sorts index first...
            lambda s: s.groupby(s).apply(lambda s: s.sort_index().tail(1)).reset_index(0, drop=True))
                                .pipe(lambda s: pd.Series(s.index, index=s.values, name='listening_region_stop_t'))
                                )

        end_listening_region.index = end_listening_region.index.rename('stim_sentcode')

        word_m_df = _word_df.merge(pd.concat([start_listening_region, end_listening_region], axis=1),
                                    left_on='stim_sentcode', right_index=True)

        #listening_start_t = word_m_df.listening_region_start_t.min()
        #listening_stop_t = word_m_df.listening_region_stop_t.max()
        self.logger.info(f"Finding closest labeled region to these stims: {self.stim_val_map}")
        nearest_stim_d = dict()
        for stim_val, stim_name in self.stim_val_map.items():
            # Get stim values for this stim_name
            _stim = stim[stim.eq(stim_val)]

            # Find the stim time (with only these values) that is nearest each words start time
            def closest_index_i_after_t(s, t, v):
                # Filter stim to times past t threshold time
                after_t_s = s[s.index >= t]
                # Within the filtered data, what i index has the value closest to v
                #closest_i_after_t_s = after_t_s.index.get_loc(v, method='nearest')
                closest_i_after_t_s = after_t_s.index.get_indexer([v], method='nearest')[0]
                # Access the index value at i
                return after_t_s.index[closest_i_after_t_s]

            stim_ixes = word_m_df.apply(lambda r: closest_index_i_after_t(_stim, r.stop_t, r.start_t), axis=1)

            # stim_ixes = _word_df.apply(lambda r: _stim.loc[_stim.index >= r.start_t].index.get_loc(r.start_t, method='nearest'), axis=1)
            nearest_stim_d[f'{stim_name}_region_start_t'] = stim_ixes
            #nearest_stim_d[f'nekarest_{stim_name}_stim_code'] = _stim.iloc[stim_ixes].values
            # nearest_stim_d[f'nearest_{stim_name}_stim_t'] = _stim.index[stim_ixes]
            #nearest_stim_d[f'{stim_name}_region_start_t'] = stim.index[stim_ixes]

        # The nearest timestamp of each key to the index (stim start of start stop word table)
        nearest_df = pd.DataFrame(nearest_stim_d, index=_word_df.index)
        nearest_df.sort_index()

        _word_nearest_df = word_m_df.join(nearest_df)

        _word_nearest_df['speaking_length_t'] = (_word_nearest_df.stop_t - _word_nearest_df.start_t)

        # for sent_code, s_df in _word_nearest_df.groupby('stim_sentcode'):
        def make_ref_offset_from_sent_code_groups(s_df):
            # how far away from start of index/ref time
            ref_diff_s = (s_df.index.to_series()
                          # Distance from the start
                          .pipe(lambda s: s - s.min())
                          #
                          .fillna(pd.Timedelta(0))
                          .rename('ref_diff_from_min'))

            # Add the difference to the 'nearest' columns - the start of those regions
            s_start_df = s_df.loc[:, nearest_df.columns].apply(lambda s: s + ref_diff_s)
            s_start_df.columns = s_start_df.columns.str.replace('region_start', 'start')

            # Determine the stop by adding the length to the new start times
            # - couldn't get this to broadcast, hence apply()
            s_stop_df = s_start_df.apply(lambda s: s + s_df['speaking_length_t'])
            s_stop_df.columns = s_start_df.columns.str.replace('start', 'stop')

            return s_start_df.join([s_stop_df, ref_diff_s])

        self.logger.info(f"Offsetting the nearest times to labeled data by the distance from start (within reference)")
        mtask_word_df = _word_nearest_df.join(_word_nearest_df
                                              .groupby('stim_sentcode')
                                              .apply(make_ref_offset_from_sent_code_groups))

        # Extract the word-level start stop for listening, assumed given in a similar cadence
        mtask_word_df['listen_start_t'] = mtask_word_df['listening_region_start_t'] + mtask_word_df['ref_diff_from_min']
        mtask_word_df['listen_stop_t'] = mtask_word_df['listen_start_t'] + mtask_word_df['speaking_length_t']

        return dict(word_start_stop_times=mtask_word_df)


@attr.s
@with_logger
class ParseSensorRAS(DictTrf):
    ras_key = attr.ib('label_contact_common')

    @staticmethod
    def ras_to_frame(ras):
        return pd.DataFrame(
            # Pass to numpy array first to have it coalesce all the scalar string arrays into a multi dim object arr
            np.array(ras),
            # hardcoded columns for now
            columns=['electrode_name', 'contact_number', 'x_coord', 'y_coord', 'z_coord']
            # Use Pandas to try and convert everything to numbers
        ).apply(pd.to_numeric, downcast='float', errors='ignore')

    def process(self, data_map):
        ras_df = self.ras_to_frame(data_map[self.ras_key])
        ras_arr = ras_df[['x_coord', 'y_coord', 'x_coord']].values
        return dict(sensor_ras_df=ras_df, sensor_ras_coord_arr=ras_arr)


@attr.s
@with_logger
class NewStimFromRegionStartStopTimes(DictTrf):
    start_t_column = attr.ib('start_t')
    stop_t_column = attr.ib('stop_t')
    label_column = attr.ib('word')
    code_column = attr.ib('word_code')

    stim_output_name = attr.ib('word_stim')

    start_stop_time_input_name = attr.ib('word_start_stop_times')
    series_with_timestamp_index = attr.ib('stim')
    default_stim_value = attr.ib(0)

    def process(self, data_map):
        _word_df = data_map[self.start_stop_time_input_name].copy()
        time_s = data_map[self.series_with_timestamp_index]
        ix = time_s.index

        output_stim = pd.Series(self.default_stim_value, index=ix, name=self.stim_output_name)

@attr.s
@with_logger
class SentenceAndWordStimFromRegionStartStopTimes(DictTrf):
    start_t_column = attr.ib('start_t')
    stop_t_column = attr.ib('stop_t')
    sent_code_column = attr.ib('stim_sentcode')
    word_code_column = attr.ib('word')
    word_stim_output_name = attr.ib('word_stim')
    word_code_map_output_name = attr.ib('word_code')
    set_as_word_stim = attr.ib(True)
    sentence_stim_output_name = attr.ib('sentence_stim')

    def process(self, data_map):
        _word_df = data_map['word_start_stop_times'].copy()
        stim = data_map['stim']
        ix = stim.index
        sentence_stim = pd.Series(0, index=ix, name='speaking_sentence_stim')
        word_stim = pd.Series(0, index=ix, name='speaking_word_stim')

        # # #
        code_maps = list()
        # We'll immediately +1 this, so we won't actually use zero for a word code - it will be silence
        working_word_ix = 0
        for i, (gname, gdf) in enumerate(_word_df.groupby(self.sent_code_column)):
            _stim = stim[stim == gname]
            sent_start_t, sent_stop_t = _stim.index.min(), _stim.index.max()
            #sent_start_t = gdf[self.start_t_column].min()
            #sent_stop_t = gdf[self.stop_t_column].max()

            #  Use index.get_indexer([item], method=...)
            start_i, stop_i = sentence_stim.index.get_indexer([sent_start_t, sent_stop_t], method='nearest')

            # Set this sentence to some incrementing indicator
            sentence_stim.iloc[start_i: stop_i] = sentence_stim.max() + 1

            # Spoken is word is all caps string
            is_word_m = gdf.word.str.upper() == gdf.word

            # Set each words region in this sentence within the word_stim
            #for ii, _start_t in enumerate(gdf[is_word_m].sort_values(self.start_t_column)[self.start_t_column].values):
                #_gdf = gdf[is_word_m].query(f"{self.word_code_column} == '{_gname}'")
            #    _gdf = gdf[is_word_m].pipe(lambda o: o[o[self.start_t_column] == _start_t])
            # _gname = _gdf[self.word_code_column].unique()
            # assert len(_gname) == 1, f"{len(_gname)} unique words in {self.word_code_column} for sentence {gname}"
            # _gname = _gname[0]

            for ii, (ix, r) in enumerate(gdf[is_word_m].sort_values(self.start_t_column).iterrows()):
                word_code_val = r[self.word_code_column]
                #_start_t = _gdf[self.start_t_column].min()
                #_stop_t = _gdf[self.stop_t_column].max()
                _start_t = r[self.start_t_column]
                _stop_t = r[self.stop_t_column]

                _start_i, _stop_i = word_stim.index.get_indexer([_start_t, _stop_t], method='nearest')
                #_start_i = sentence_stim.index.get_loc(_start_t, method='nearest')
                #_stop_i = sentence_stim.index.get_loc(_stop_t, method='nearest')

                word_code = (working_word_ix := working_word_ix + 1)
                code_maps.append({self.sent_code_column: gname, self.word_code_column: word_code_val,
                                  self.word_code_map_output_name: word_code, 'start_t': _start_t})
                word_stim.iloc[_start_i: _stop_i] = word_code

        out = {self.word_stim_output_name: word_stim,
               self.sentence_stim_output_name: sentence_stim,
                #'wsst_df': wsst_df
                }

        if self.set_as_word_stim:
            code_df = pd.DataFrame(code_maps)
            wsst_df = _word_df.merge(code_df, on=['start_t', 'stim_sentcode', self.word_code_column], how='left')
            wsst_df = wsst_df.set_index('start_t', drop=False).rename_axis(index='start_time')
            wsst_df[self.word_code_map_output_name] = wsst_df[self.word_code_map_output_name].fillna(0).astype(int)
            word_code_d = wsst_df.set_index('word_code').drop(0).word.to_dict()
            out.update(
                {'word_code_frame': code_df,
                 'word_start_stop_times': wsst_df,
                 'word_code_d': word_code_d}
            )

        return out


def object_as_key_or_itself(key_or_value, remap=None):
    """
    Returns a value from (in order):
        - remap[key_or_value]
        - remap
        - key_or_value
    """
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
    stim_pre_process_f = attr.ib(None)

    def process(self, data_map):
        return self.make_sample_indices(data_map[self.stim_key], data_map[self.fs_key], win_size=self.window_size,
                                        index_shift=self.index_shift,
                                        stim_target_value=self.stim_target_value, stim_value_remap=self.stim_value_remap,
                                        existing_sample_indices_map=data_map.get('sample_index_map'),
                                        stim_pre_process_f=self.stim_pre_process_f if self.stim_pre_process_f is not None
                                                            else lambda _stim: _stim)

    @classmethod
    def make_sample_indices(cls, stim, fs, win_size,
                            index_shift, stim_target_value, stim_value_remap,
                            existing_sample_indices_map,
                            stim_pre_process_f):
        index_shift = pd.Timedelta(0, 's') if index_shift is None else index_shift
        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        sample_indices = dict()
        expected_window_samples = int(fs * win_size.total_seconds())

        target_indexes = (stim.pipe(stim_pre_process_f) == stim_target_value).pipe(lambda s: s[s].index.tolist())
        target_indices = [stim.loc[offs + index_shift:offs + win_size + index_shift].iloc[:expected_window_samples].index
                          for offs in target_indexes
                          if len(stim.loc[offs + index_shift:offs + win_size + index_shift]) >= expected_window_samples]

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
                                        max_target_region_size=self.max_target_region_size,
                                        existing_sample_indices_map=data_map.get('sample_index_map'),
                                        existing_indices_sources_map=data_map.get('index_source_map'),
                                        stim_value_remap=self.stim_value_remap,
                                        source_name=self.stim_key)

    @classmethod
    def make_sample_indices(cls, stim, fs, win_size, target_onset_ref, target_onset_shift,
                            target_offset_ref, target_offset_shift,
                            #silence_value, silence_samples, silent_window_scale,
                            max_target_region_size, existing_sample_indices_map=None,
                            existing_indices_sources_map=None,
                            stim_value_remap=None,
                            source_name=None):

        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        existing_indices_sources_map = dict() if existing_indices_sources_map is None else existing_indices_sources_map

        expected_window_samples = int(fs * win_size.total_seconds())
        #label_region_sample_size = int(fs * label_region_size.total_seconds())
        cls.logger.info((fs, win_size))
        cls.logger.info("Samples per window: %d" % expected_window_samples)

        # Will map of codes to list of indices into the stim signal:
        # word_code->List[pd.Index, pd.Index, ...]
        sample_indices = dict()
        indices_sources = dict()

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
            s_ix = stim[target_start_t:target_stop_t].index
            assert s_ix.is_unique, f"Index between {target_start_t} and {target_stop_t} is not unique!"
            target_start_ixes = s_ix.tolist()#[:-expected_window_samples]

            # Go through the labeled region indices and pull a window of data
            target_indices = [stim.loc[offs:offs + win_size].iloc[:expected_window_samples].index
                                for offs in target_start_ixes[:max_target_region_size]
                                    if len(stim.loc[offs:offs + win_size]) >= expected_window_samples]

            stim_key = object_as_key_or_itself(stim_value, stim_value_remap)
            sample_indices[stim_key] = sample_indices.get(stim_key, list()) + target_indices
            indices_sources[stim_key] = indices_sources.get(stim_key, source_name)

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

        if existing_indices_sources_map is not None:
            existing_indices_sources_map.update(indices_sources)
            indices_sources = existing_indices_sources_map

        return dict(sample_index_map=sample_indices, n_samples_per_window=expected_window_samples,
                    index_source_map=indices_sources)


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
import attr


@attr.s
class DictTrf(BaseEstimator, TransformerMixin):
    def transform(self, data_map):
        updates = self.process(data_map)
        data_map.update(updates)
        return data_map

    def process(self, data_map):
        raise NotImplementedError()


@attr.s
class SubsampleSignal(DictTrf):
    signal_keys = attr.ib(('ecog', 'stim', 'stim_diff'))
    signal_rate_key = attr.ib('fs_signal')
    rate = attr.ib(2)

    def process(self, data_map):
        output = {k: data_map[k].iloc[::self.rate] for k in self.signal_keys}
        output[self.signal_rate_key] = int((1. / self.rate) * data_map[self.signal_rate_key])
        return output


@attr.s
class PowerThreshold(DictTrf):
    speaking_threshold = attr.ib(0.007)
    silence_threshold = attr.ib(0.001)

    speaking_window_samples = attr.ib(48000)
    # More silence data, so require larger region of threshold check
    silence_window_samples = attr.ib(96000)

    def process(self, data_map):
        return self.power_threshold(data_map['audio'], data_map['stim'],
                                    speaking_threshold=self.speaking_threshold,
                                    speaking_window_samples=self.speaking_window_samples,
                                    silence_threshold=self.silence_threshold,
                                    silence_window_samples=self.silence_window_samples)

    @staticmethod
    def power_threshold(audio_s, stim_s, speaking_threshold,
                        speaking_window_samples,
                        silence_threshold,
                        silence_window_samples):
        rolling_pwr = (audio_s
                       .abs().rolling(speaking_window_samples, center=True)
                       .max().reindex(stim_s.index, method='nearest').fillna(0))

        speaking_stim_auto_m = (stim_s != 0.) & (rolling_pwr > speaking_threshold)

        silence_rolling_pwr = (audio_s
                               .abs().rolling(silence_window_samples, center=True)
                               .max().reindex(stim_s.index, method='nearest').fillna(0))
        silence_stim_auto_m = (stim_s == 0.) & (~speaking_stim_auto_m) & (silence_rolling_pwr < silence_threshold)

        # Is the number of unique word codes different when using the threshold selected subset we
        # just produced (stim_auto_m)?
        # - Subtract one for no speech (0)
        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[speaking_stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[speaking_stim_auto_m].nunique(False))
            print(msg)

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

    @staticmethod
    def make_sample_indices(stim, fs, win_size, target_onset_ref, target_onset_shift,
                            target_offset_ref, target_offset_shift,
                            #silence_value, silence_samples, silent_window_scale,
                            max_target_region_size, existing_sample_indices_map=None, stim_value_remap=None):

        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        expected_window_samples = int(fs * win_size.total_seconds())
        #label_region_sample_size = int(fs * label_region_size.total_seconds())
        print((fs, win_size))
        print("Samples per window: %d" % expected_window_samples)

        # Will map of codes to list of indices into the stim signal:
        # word_code->List[pd.Index, pd.Index, ...]
        sample_indices = dict()

        # TODO: This will not work for constant stim value (i.e. True/False, 1/0)?
        # TODO: Need to review UCSD data and how to write something that will work for its regions
        s_grp = stim[stim > 0].pipe(lambda _s: _s.groupby(_s))
        for stim_value, g_s in tqdm(s_grp):
            start_t = g_s.index.min()
            stop_t = g_s.index.max()

            if target_onset_ref == 'rising':
                target_start_t = start_t + target_onset_shift
            elif target_onset_ref == 'falling':
                target_start_t = stop_t + target_onset_shift
            else:
                raise ValueError(f"Dont understand {target_onset_ref}")

            if target_offset_ref == 'rising':
                speaking_stop_t = start_t + target_offset_shift
            elif target_offset_ref == 'falling':
                speaking_stop_t = stop_t + target_offset_shift
            else:
                raise ValueError(f"Dont understand {target_offset_ref}")

            # Get the window starting indices for each region of interest
            # Note on :-expected_window_samples
            #   - this removes the last windows worth since windows starting here would have out of label samples
            target_start_ixes = stim[target_start_t:speaking_stop_t].index.tolist()[:-expected_window_samples]

            # Go through the labeled region indices and pull a window of data
            target_indices = [stim.loc[offs:offs + win_size].iloc[:expected_window_samples].index
                                for offs in target_start_ixes[:max_target_region_size]]
            if isinstance(stim_value_remap, dict):
                stim_key = stim_value_remap[stim_value]
            elif stim_value_remap is not None:
                stim_key = stim_value_remap
            elif stim_value_remap is None:
                stim_key = stim_value
            else:
                raise ValueError(f"Dont know how to handle stim_value_remap of type: {type(stim_value_remap)}")

            sample_indices[stim_key] = sample_indices.get(stim_key, list()) + target_indices

        # Go through all samples - make noise if sample size is off (or should throw error?)
        for k, _s in sample_indices.items():
            for i, _ixs in enumerate(_s):
                if len(_ixs) != expected_window_samples:
                    print(f"[{k}][{i}] ({len(_ixs)}): {_ixs}")

        # Debug code printing the unique lengths of each window for each word code
        #print({k : sorted(list(set(map(len, _s)))) for k, _s in sample_indices.items()})
        if existing_sample_indices_map is not None:
            existing_sample_indices_map.update(sample_indices)
            sample_indices = existing_sample_indices_map
        return dict(sample_index_map=sample_indices)


@attr.s
class SampleIndicesFromStimV2(DictTrf):
    stim_key = attr.ib('stim')
    fs_key = attr.ib('fs_signal')
    window_size = attr.ib(pd.Timedelta(0.5, 's'))

    # One of rising or falling
    speaking_onset_reference = attr.ib('rising')
    speaking_offset_reference = attr.ib('falling')
    speaking_onset_shift = attr.ib(pd.Timedelta(-0.50, 's'))
    speaking_offset_shift = attr.ib(pd.Timedelta(0., 's'))

    max_speaking_region_size = attr.ib(600)

    silence_stim_value = attr.ib(0)
    silence_samples = attr.ib(None)
    silent_window_scale = attr.ib(4)

    def process(self, data_map):
        return self.make_sample_indices(data_map[self.stim_key], data_map[self.fs_key],
                                        win_size=self.window_size,
                                        speaking_onset_ref=self.speaking_onset_reference,
                                        speaking_onset_shift=self.speaking_onset_shift,
                                        speaking_offset_ref=self.speaking_offset_reference,
                                        speaking_offset_shift=self.speaking_offset_shift,
                                        silence_value=self.silence_stim_value, silence_samples=self.silence_samples,
                                        silent_window_scale=self.silent_window_scale,
                                        max_speaking_region_size=self.max_speaking_region_size)

    @staticmethod
    def make_sample_indices(stim, fs, win_size, speaking_onset_ref, speaking_onset_shift,
                            speaking_offset_ref, speaking_offset_shift, silence_value, silence_samples,
                            silent_window_scale, max_speaking_region_size):

        expected_window_samples = int(fs * win_size.total_seconds())
        #label_region_sample_size = int(fs * label_region_size.total_seconds())
        print((fs, win_size))
        print("Samples per window: %d" % expected_window_samples)

        # Will map of codes to list of indices into the stim signal:
        # word_code->List[pd.Index, pd.Index, ...]
        sample_indices = dict()

        # TODO: This will not work for constant stim value (i.e. True/False, 1/0)?
        # TODO: Need to review UCSD data and how to write something that will work for its regions
        s_grp = stim[stim > 0].pipe(lambda _s: _s.groupby(_s))
        for stim_value, g_s in tqdm(s_grp):
            start_t = g_s.index.min()
            stop_t = g_s.index.max()

            if speaking_onset_ref == 'rising':
                speaking_start_t = start_t + speaking_onset_shift
            elif speaking_onset_ref == 'falling':
                speaking_start_t = stop_t + speaking_onset_shift
            else:
                raise ValueError(f"Dont understand {speaking_onset_ref}")

            if speaking_offset_ref == 'rising':
                speaking_stop_t = start_t + speaking_offset_shift
            elif speaking_offset_ref == 'falling':
                speaking_stop_t = stop_t + speaking_offset_shift
            else:
                raise ValueError(f"Dont understand {speaking_offset_ref}")

            # Get the window starting indices for each region of interest
            # Note on :-expected_window_samples
            #   - this removes the last windows worth since windows starting here would have out of label samples
            speaking_start_ixes = stim[speaking_start_t:speaking_stop_t].index.tolist()[:-expected_window_samples]

            # Go through the labeled region indices and pull a window of data
            speaking_indices = [stim.loc[offs:offs + win_size].iloc[:expected_window_samples].index
                                for offs in speaking_start_ixes[:max_speaking_region_size]]
            sample_indices[stim_value] = speaking_indices

        silence_m = (stim == silence_value)
        # Find regions four times the size of the label regions that are completely silent
        # values are the center of the silent regions
        speaking_samp_count = (~silence_m).rolling(silent_window_scale * win_size, center=True).sum().dropna()
        # Ignore a windows worth at the start and end
        speaking_samp_count = speaking_samp_count.loc[win_size: speaking_samp_count.index.max() - win_size]
        # Interested in areas where no speaking was detected within the scaled window
        silence_center_s = speaking_samp_count[speaking_samp_count.eq(0)]

        print(f"Max window samples: {expected_window_samples}")
        # If the number of samples to take is not provided, then take the same number as there were positive samples
        if silence_samples is None:
            n_pos_samples = sum(len(_ix) for _ix in sample_indices.values())
            print(f"N pos samples: {n_pos_samples}")
            print(f"N silence centers: {len(silence_center_s)}")
            silence_samples = min(n_pos_samples, len(silence_center_s))
            print(f"Taking {silence_samples} silence samples")

        # Shift the centers by half the window size - placing the index at the
        # start of the window to be extracted rather than the center
        _centers_s = silence_center_s.sample(silence_samples).index
        #_offs_s = _centers_s + (win_size * silent_window_scale / 2)
        # Move back half the real window size so it's centered when it's extracted from left to right
        # (from the leftmost offset)
        _offs_s = _centers_s - (win_size / 2)

        # Go through the labeled region indices and pull a window of data
        silence_indices = [stim.loc[offs:offs + win_size].iloc[:expected_window_samples].index
                           # SKip over anything not large enough for a window
                           for offs in _offs_s if len(stim.loc[offs:]) > expected_window_samples]
        sample_indices[silence_value] = silence_indices

        # Go through all samples - make noise if sample size is off (or should throw error?)
        for k, _s in sample_indices.items():
            for i, _ixs in enumerate(_s):
                if len(_ixs) != expected_window_samples:
                    print(f"[{k}][{i}] ({len(_ixs)}): {_ixs}")

        # Debug code printing the unique lengths of each window for each word code
        #print({k : sorted(list(set(map(len, _s)))) for k, _s in sample_indices.items()})

        return dict(sample_index_map=sample_indices)


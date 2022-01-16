import numpy as np
import pandas as pd
from scipy import signal as sig
import attr


def peak_normalization(arr_s, output_type=int):
    """
    Divide by max value present in array, multiply by the datatypes max, convert to integer
    :param arr_s:
    :return:
    """
    return ((arr_s / arr_s.max()) * np.iinfo(arr_s.dtype).max).round().astype(output_type)


def compute_speaking_mask(wav, power_win_size=1024,
                          avg_power_thresh=50, avg_power_win_size=4096):
    """
    Uses rolling power estimate to gate the input - 1 or True is passthrough
    """
    pwr_s = wav.abs().rolling(power_win_size).mean()
    return (pwr_s > avg_power_thresh).rolling(avg_power_win_size).max() > 0


def speech_region_index_from_speech_mask(spk_mask_s):
    leading_edge_s = ((spk_mask_s.diff().replace(0, np.nan) + 1) / 2)
    word_trail_label_s = leading_edge_s.fillna(0).cumsum()
    word_index = spk_mask_s * word_trail_label_s

    # Incrementing negatives during silence, 0 everywhere else
    neg_neutral_cnt = -(word_trail_label_s - word_index - spk_mask_s) - 1

    # combine to make incrementing negatives identify leading silence regions
    # and incrementing positive labels for speech
    r_ix = pd.Series(np.where(neg_neutral_cnt == 0, word_index, neg_neutral_cnt),
                     index=word_index.index)
    return r_ix


def compute_speech_index(wav, speaking_mask_f=compute_speaking_mask):
    """
    Provide a function that
    """
    spk_mask_s = speaking_mask_f(wav).astype(int)
    return speech_region_index_from_speech_mask(spk_mask_s)


######
def fft(data, fs=1000):
    fft_vals = np.absolute(np.fft.rfft(data, axis=0))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
    #return fft_freq, fft_vals
    s = pd.Series(fft_vals, index=fft_freq, name='fft_values')
    s.index.name = 'hz'
    return s


def filter(s, band, sfreq=1000, verbose=False, n_jobs=4,
           method='fir'):
    try:
        import mne
    except ImportError:
        print("Need MNE package to filter")
        raise
    if isinstance(s, pd.Series):
        _s = s.values
    elif isinstance(s, pd.DataFrame):
        _s = s.values.T
    else:
        _s = s

    _s = _s.astype('float64')

    filtered_phase_arr = mne.filter.filter_data(_s, sfreq,
                                                *band,
                                                verbose=verbose,
                                                method=method,
                                                n_jobs=n_jobs)

    if isinstance(s, pd.Series):
        ret = pd.Series(filtered_phase_arr, index=s.index, name=s.name)

    elif isinstance(s, pd.DataFrame):
        ret = pd.DataFrame(filtered_phase_arr.T,
                           columns=s.columns,
                           index=s.index)
    else:
        ret = filtered_phase_arr

    return ret


def make_hilbert_df(x):
    """
    Extracts analytic signal and stores components into a DataFrame
    - Real and imaginary components
    - Angle, both wrapped and unwrapped

    :param x:
    :return:
    """
    x_hil = sig.hilbert(x)

    x_hil_real = x_hil.real
    x_hil_imag = x_hil.imag

    x_hil_angle = np.angle(x_hil)
    x_hil_phase = np.unwrap(x_hil_angle)

    hilbert_df = pd.DataFrame(dict(z_t_real=x_hil_real, z_t_imag=x_hil_imag,
                                   z_t_angle=x_hil_angle, z_t_unwrap=x_hil_phase,
                                   envelope=np.abs(x_hil)
                                   ),
                              index=x.index)
    hilbert_df['signal'] = x
    return hilbert_df



###
@attr.s
class ProcessPipeline:
    """
    Collection of ProcessSteps to be run in a sequence
    """
    name = attr.ib(None)
    steps = attr.ib(None)

    def __attrs_post_init__(self):
        if self.steps is None:
            self.steps = self.make_steps()

        if self.name is None:
            self.name = self.__class__.__name__

    def __call__(self, x):
        _x = x
        for i, (s_func, s_kws) in enumerate(self.steps):
            _x = s_func(_x, **s_kws)
        return _x

    def __rshift__(self, other):

        if isinstance(other, ProcessStep):
            self.steps.append((other, dict()))
            ret_o = self
        elif isinstance(other, ProcessPipeline):
            ret_o = ProcessPipeline(name=f"{self.name}__{other.name}", steps=self.steps + other.steps)
        else:
            raise ValueError("Only ProcessStep/Pipeline types allowed for now :|")

        return ret_o


@attr.s
class ProcessStep:
    """
    Perform arbitrary processing on data_map in abstract `step` method. Support
    remapping inputs/outputs (i.e. x[k] = x[v] from k,v in d.items()) and
    overidding an entry before passing to step.

    (TODO: How to add params docs in attrs? Seems to be an open issue...
    Parameters
    ----------
    name : str or None (default=None)
        Friendly/display name of the process step
    input_override : dict or None (default=None)
        key-value pairs to replace in the input key-value pairs before being passed to step
    input_remap : dict or None (default=None)
        key-value pairs (k, v), set value at k to value at v in the input map
    expects : list or None (default=None)
        List of keys expected to be found in the input data mapping
    outputs : list or None (default=None)
        List of keys this process will add/modify in the data map

    """
    name = attr.ib(None)


    input_override = attr.ib(None)
    input_remap = attr.ib(None)
    output_remap = attr.ib(None)

    expects = None
    outputs = None

    def __attrs_post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    def __call__(self, x):
        if self.input_remap is not None:
            for k, v in self.input_remap.items():
                x[k] = x[v]

        if self.input_override is not None:
            for k, v in self.input_override.items():
                x[k] = v

        for e in self.expects:
            if e not in x:
                raise ValueError(f"Missing input field: {e} (all fields: {', '.join(x.keys())}")

        x_updates = self.step(x)
        # TODO: maybe deep copy? add parameter for clone?
        _x = x
        for k, v in x_updates.items():
            if k not in self.outputs:
                raise ValueError(f"Unspecified output {k}")
            _x[k] = v

        #_x = x
        #for i, (s_func, s_kws) in enumerate(self.steps):
        #    _x = s_func(_x, **s_kws)

        if self.output_remap is not None:
            print("--")
            for k, v in self.output_remap.items():
                print(f"remap: {k} <- {v}")
                _x[k] = _x[v]

        if self.outputs is not None:
            for o in self.outputs:
                if o not in _x:
                    raise ValueError(f"Missing output field: {e} (all fields: {', '.join(x.keys())}")

        return _x

    def __rshift__(self, other):
        return ProcessPipeline(steps=[(self, dict()), (other, dict())])

    def step(self, data_map):
        raise NotImplementedError()


class WordStopStartTimeMap(ProcessStep):
    verbose = attr.ib(False)
    expects = ['stim', 'stim_diff', 'word_code_d']
    outputs = ['start_times_d', 'stop_times_d']

    def step(self, data_map):
        return self.default_preprocessing(data_map['stim_diff'],
                                          data_map['word_code_d'],
                                          self.verbose)

    @staticmethod
    def default_preprocessing(stim_diff_s, word_code_d, verbose=False):
        #stim_s = data_map['stim']
        #stim_diff_s = data_map['stim_diff']
        ######
        # Stim event codes and txt
        # 0 is neutral, so running difference will identify the onsets
        # stim_diff_s = stim_s.diff().fillna(0).astype(int)
        # return dict(stim_diff=stim_diff_s)

        #word_code_d = data_map['word_code_d']
        ######
        # Stim event codes and txt
        # 0 is neutral, so running difference will identify the onsets
        # stim_diff_s = stim_s.diff().fillna(0).astype(int)

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

        updates = dict(  # stim_diff=stim_diff_s,
            # stim_auto=stim_auto_s,
            # stim_auto_diff=stim_auto_diff_s,
            start_times_d=start_time_map,
            stop_times_d=stop_time_map)
        # data_map.update(updates)
        return updates


@attr.s
class PowerThreshold(ProcessStep):
    """
    Produce a new stim signal from audio levels peak amplitude over N samples

    Take the max absolute value of audio over `window_samples` sample rolling window.
    Produces step signal, positive 1 where this value is over the threshold *and*
    the stim signal is not 0.
    """
    name = 'threshold'

    #input_override = attr.ib(None)
    threshold = attr.ib(0.007)
    window_samples = attr.ib(48000)

    #input_remap = attr.ib(None)#attr.Factory(lambda :dict(stim='stim_pwrt', stim_diff='stim_pwrt_diff')))
    output_remap = attr.ib(attr.Factory(lambda: dict(stim='stim_pwrt', stim_diff='stim_pwrt_diff')))

    expects = ['audio', 'stim']
    outputs = ['stim_pwrt', 'stim_pwrt_diff', 'rolling_audio_pwr']

    def step(self, data_map):
        return self.stim_adjust__power_threshold(data_map['audio'], data_map['stim'], threshold=self.threshold,
                                                 window_samples=self.window_samples)

    @staticmethod
    def stim_adjust__power_threshold(audio_s, stim_s, threshold=0.007, window_samples=48000):
        #audio_s = data_map['audio']
        #stim_s = data_map['stim']

        rolling_pwr = audio_s.abs().rolling(window_samples, center=True).max().reindex(stim_s.index, method='nearest').fillna(0)
        #rolling_pwr = (rolling_pwr
        #            .reindex(rolling_pwr.index.union(stim_s.index))
        #            .interpolate(method='time')
        #            .reindex(stim_s.index)
        #            )
        #rolling_pwr = rolling_pwr.shift(- (window_samples // 2)).fillna(0)
        stim_auto_m = (stim_s != 0.) & (rolling_pwr > threshold)

        # Is the number of unique word codes different when using the threshold selected subset we
        # just produced (stim_auto_m)?
        # - Subtract one for no speech (0)
        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[stim_auto_m].nunique(False))
            print(msg)

        # Create a new stim array with original word code where it's set, otherwise zero
        stim_pwrt_s = pd.Series(np.where(stim_auto_m, stim_s, 0), index=stim_s.index)
        stim_pwrt_diff_s = stim_pwrt_s.diff().fillna(0).astype(int)

        updates = dict(stim_pwrt=stim_pwrt_s, stim_pwrt_diff=stim_pwrt_diff_s,
                       rolling_audio_pwr=rolling_pwr)
        return updates

@attr.s
class PowerQuantile(ProcessStep):
    """
    Produce a new stim signal from audio levels. Takes the mean absolute value of a rolling window,
    then sets the new stim signal where both the audio mean absolute value is above the chosen
    quantile level and the original stim code is set.
    """
    q = attr.ib(0.75)
    trim_sample_n = attr.ib(50)

    output_remap = attr.ib(attr.Factory(lambda :dict(stim='stim_pwrq', stim_diff='stim_pwrq_diff')))
    expects = ['audio', 'stim']
    outputs = ['stim_pwrq', 'stim_pwrq_diff', 'rolling_audio_pwr']

    def step(self, data_map):
        return self.stim_adjust__power_quantile(data_map['audio'], data_map['stim'],
                                                self.q, self.trim_sample_n)

    @staticmethod
    def stim_adjust__power_quantile(audio_s, stim_s, q=0.75, trim_sample_n=50):
        #audio_s = data_map['audio']
        #stim_s = data_map['stim']

        #rolling_pwr = audio_s.abs().rolling(48000).mean().reindex(stim_s.index).fillna(method='ffill').fillna(0)
        rolling_pwr = audio_s.abs().rolling(48000).mean(center=True).reindex(stim_s.index, method='nearest').fillna(0)

        # TODO: before taking quantile or grabbing slice, clip ends to avoid positioning at extreme of stim code region
        #q_thresh_s = rolling_pwr.groupby(stim_s).quantile(q)
        q_thresh_s = rolling_pwr.groupby(stim_s).apply(lambda s: s.iloc[trim_sample_n:-trim_sample_n].quantile(q))
        # Label zero doesn't really matter, but set it's threshold to zero
        q_thresh_s.loc[0] = 0

        # Create a mask of regions where stim code is set and the audio power is above
        # quantile(q) for the stim region
        stim_auto_m = (stim_s != 0.) & (rolling_pwr >= stim_s.map(q_thresh_s.to_dict()))

        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[stim_auto_m].nunique(False))
            print(msg)

        # New stim takes origin stim value only where the mask is set, otherwise 0
        stim_pwrq_s = pd.Series(np.where(stim_auto_m, stim_s, 0), index=stim_s.index)

        # Diff it: positive stim value indicates start, negative stim value
        # denotes beginning of of the end of the word
        stim_pwrq_diff_s = stim_pwrq_s.diff().fillna(0).astype(int)

        updates = dict(stim_pwrq=stim_pwrq_s, stim_pwrq_diff=stim_pwrq_diff_s,
                       rolling_audio_pwr=rolling_pwr)
        #remaps = dict(stim='stim_pwrq', stim_diff='stim_pwrq_diff')
        #return updates, remaps
        return updates


@attr.s
class SubsampleECOG(ProcessStep):
    """
    Take every `rate` sample - e.g. if rate == 2, every other sample is kept, halving rate. Doesn't
    do any interpolation, so must be whole number step sizes.
    """
    rate = attr.ib(2)
    expects = ['ecog', 'fs_signal', 'stim', 'stim_diff']
    outputs = ['ecog_subs', 'fs_signal_subs', 'stim_subs', 'stim_diff_subs']

    output_remap = attr.ib(attr.Factory(lambda : dict(ecog='ecog_subs', fs_signal='fs_signal_subs',
                                                      stim='stim_subs', stim_diff='stim_diff_subs')))

    def step(self, data_map):
        x = data_map['ecog']
        #fs_signal = getattr(self, 'fs_signal', 1. / self.rate)
        return dict(ecog_subs=x.iloc[::self.rate, ],
                    stim_subs=data_map['stim'].iloc[::self.rate,],
                    stim_diff_subs=data_map['stim_diff'].iloc[::self.rate,],
                    fs_signal_subs=int((1. / self.rate) * data_map['fs_signal']))


@attr.s
class SampleIndicesFromStim(ProcessStep):
    """
    Produces a mapping of stim_codes (word ids) to a list of window indices into the source data.

    Expects values above 0 to True positives, with different values for different stimulus in continuous regions.
    i.e. this probably won't work when a stim code can be separated/interleaved with other different codes.
    """
    expects = ['stim_diff']
    outputs = ['sample_index_map']
    ecog_window_size = attr.ib(600)
    ecog_window_n = attr.ib(60)
    ecog_window_step_sec = attr.ib(pd.Timedelta(0.01, 's'))
    ecog_window_shift_sec = attr.ib(pd.Timedelta(0.75, 's'))

    def step(self, data_map):
        return self.make_sample_indices(data_map['stim_diff'], self.ecog_window_size, self.ecog_window_n,
                                        self.ecog_window_step_sec, self.ecog_window_shift_sec)

    @staticmethod
    def make_sample_indices(label_index, win_size, ecog_window_n, ecog_window_step_sec, ecog_window_shift_sec):
        """
        Group by a label index (repeated label measures, one per sample)
        to find start and end points, then slice at the appropriate offsets
        back into the original data to capture the index (not the data).


        :param label_index:
        :param win_size:
        :param win_step:
        :return:
        """
        #label_index = data_map['stim_diff']

        sample_indices = dict()
        ####
        # Separate into two sets and loops so that the negative
        # samples can be derived from the positive samples
        pos_ix = label_index[label_index > 0]
        neg_ix = label_index[label_index < 0]
        pos_grp = pos_ix.groupby(pos_ix)
        neg_grp = neg_ix.groupby(neg_ix)

        # Positive windows extracted first - this way negative windows
        # can potentially be extracted based on the positive windows
        for wrd_id, wave_wrd_values in pos_grp:
            start_t = wave_wrd_values.index.min() - ecog_window_shift_sec

            sample_indices[wrd_id] = [label_index
                                        # Filter the start up to start time + all previous steps
                                        .loc[start_t + i * ecog_window_step_sec:]
                                        # Take the next N samples to get correct window size
                                        .iloc[:win_size]
                                        # only store the index, not all the data
                                        .index
                                      for i in range(ecog_window_n)]

        for wrd_id, wave_wrd_values in neg_grp:
            start_t = wave_wrd_values.index.min()  # - cls.ecog_window_shift_sec

            # Where the last positive window ends
            pos_ix = sample_indices[-wrd_id][-1].max()
            # Start from the end of positive window if it overlaps our current start time
            # Should only be relevant for first window
            if start_t < pos_ix:
                start_t = pos_ix

            sample_indices[wrd_id] = [label_index
                                          .loc[start_t + i * ecog_window_step_sec:]
                                          .iloc[:win_size]
                                          .index
                                      for i in range(ecog_window_n)]

        # Hacky, but just remove anything that's not the right size
        # sample_indices[wrd_id] = [ixes for ixes in sample_indices[wrd_id]
        #                          if len(ixes) == win_size]
        sample_indices = {w: [ix for ix in ixes if len(ix) == win_size]
                          for w, ixes in sample_indices.items()}
        for w, ixes in sample_indices.items():
            if len(ixes) == 0:
                msg = ("Error: word %d had no windows" % w)
                print(msg)

        # TODO: Compute flat_index_map and flat_keys and return

        return dict(sample_index_map=sample_indices)


#import torchaudio
@attr.s
class MFCC(ProcessStep):
    # Ecog is used to reindex the MFCC output so it aligns
    expects = ['audio', 'fs_audio', 'ecog']
    outputs = ['mfcc']
    num_mfcc = attr.ib(13)

    #def __attrs_post_init__(self):
    #    import torchaudio
    #    self.mfcc_m = torchaudio.transforms.MFCC()
    def step(self, data_map):
        return dict(mfcc=self.aligned_ecog_mfcc(self.num_mfcc, data_map['ecog'].index, data_map['audio'], data_map['fs_audio']))

    @staticmethod
    def aligned_ecog_mfcc(num_mfcc, ix_to_align, audio_s, audio_rate):
        from python_speech_features import mfcc
        #sig = data_map['audio']
        #rate = data_map['fs_audio']
        winstep = 1 / 200
        winlen = 0.02
        nfft = 1024

        mfcc_feat = mfcc(audio_s, audio_rate, winlen=winlen, winstep=winstep, nfft=nfft,
                         numcep=num_mfcc)

        max_t = ix_to_align.max()
        mfcc_sample_rate = mfcc_feat.shape[0] / max_t.total_seconds()

        ix = pd.TimedeltaIndex(pd.RangeIndex(0, mfcc_feat.shape[0]) / mfcc_sample_rate, unit='s')

        mfcc_df = pd.DataFrame(mfcc_feat, index=ix, columns=[f"mfcc{i}" for i in range(mfcc_feat.shape[-1])])

        reix_mfcc_df = mfcc_df.reindex(ix_to_align,
                                       tolerance=pd.Timedelta(15, unit='ms'),
                                       method='nearest')
        return reix_mfcc_df
        #return dict(mfcc=reix_mfcc_df)

        #self.mfcc_f = getattr(self, 'mfcc_f',
        #                      torchaudio.transforms.MFCC(data_map['fs_audio'],
        #                                                 self.num_mfcc))
        #return dict(mfcc=self.mfcc_f(data_map['audio']))


@attr.s
class SampleIndicesFromStimV2(ProcessStep):
    #expects = ['start_times_d', 'fs_signal']
    expects = ['stim_diff', 'fs_signal']
    outputs = ['sample_index_map']

    window_size = attr.ib(pd.Timedelta(0.5, 's'))

    # One of rising or falling
    speaking_onset_reference = attr.ib('rising')
    speaking_offset_reference = attr.ib('falling')
    speaking_onset_shift = attr.ib(pd.Timedelta(-0.50, 's'))
    speaking_offset_shift = attr.ib(pd.Timedelta(0., 's'))

    silence_stim_value = attr.ib(0)
    silence_samples = attr.ib(None)

    def step(self, data_map):
        return self.make_sample_indices(data_map, win_size=self.window_size,
                                        speaking_onset_ref=self.speaking_onset_reference, speaking_onset_shift=self.speaking_onset_shift,
                                        speaking_offset_ref=self.speaking_offset_reference, speaking_offset_shift=self.speaking_offset_shift,
                                        silence_value=self.silence_stim_value, silence_samples=self.silence_samples)

    @staticmethod
    def make_sample_indices(data_map, win_size, speaking_onset_ref, speaking_onset_shift,
                            speaking_offset_ref, speaking_offset_shift, silence_value, silence_samples=None):
        from tqdm.auto import tqdm
        label_index = data_map['stim_diff']
        fs = data_map['fs_signal']
        stim = data_map['stim']

        max_window_samples = int(fs * win_size.total_seconds())
        #label_region_sample_size = int(fs * label_region_size.total_seconds())
        print((fs, win_size))
        print("Max window size: %d" % max_window_samples)

        sample_indices = dict()

        # TODO: Need to review UCSD data and how to write something that will work for its regions
        #raise NotImplementedError()

        s_grp = stim[stim > 0].pipe(lambda _s: _s.groupby(_s))
        for gname, g_s in tqdm(s_grp):
            start_t = g_s.index.min()
            stop_t = g_s.index.max()

            if speaking_onset_ref == 'rising':
                speaking_start_t = start_t + speaking_onset_shift
            elif speaking_onset_ref == 'falling':
                speaking_start_t = stop_t + speaking_onset_shift
            else:
                raise ValueError()

            if speaking_offset_ref == 'rising':
                speaking_stop_t = start_t + speaking_offset_shift
            elif speaking_offset_ref == 'falling':
                speaking_stop_t = stop_t + speaking_offset_shift
            else:
                raise ValueError()

            # Get the window starting indices for each region of interest
            # Note on :-max_window_samples - this removes the last windows worth since windows starting here would have out of label samples
            speaking_start_ixes = (stim[speaking_start_t:speaking_stop_t]  # .iloc[:label_region_sample_size]
                                       .index.tolist()[:-max_window_samples])

            # Go through the labeled region indices and pull a window of data
            speaking_indices = [label_index.loc[offs:offs + win_size].index
                                for offs in speaking_start_ixes]
            sample_indices[gname] = speaking_indices

        silence_m = (stim == silence_value)
        # Find regions twice the size of the label regions that are completely silent
        # values are the center of the silent regions
        not_silent_samp_count = (~silence_m).rolling(2*win_size, center=True).sum()
        silence_center_s = not_silent_samp_count[not_silent_samp_count.eq(0)]

        # If the number of samples to take is not provided, then take the same number as there were positive samples
        if silence_samples is None:
            n_pos_samples = sum(len(_ix) for _ix in sample_indices.values())
            print(f"N pos samples: {n_pos_samples}")
            print(f"N silence centers: {len(silence_center_s)}")
            silence_samples = min(n_pos_samples, len(silence_center_s))
            print(f"Taking {silence_samples} silence samples")

        # Shift the centers by half the window size - placing the window to be extracted
        _centers_s = silence_center_s.sample(silence_samples).index
        _centers_s = _centers_s - (win_size / 2)

        # Go through the labeled region indices and pull a window of data
        silence_indices = [stim.loc[offs:offs + win_size].index
                            for offs in _centers_s]
        sample_indices[silence_value] = silence_indices

        return dict(sample_index_map=sample_indices)


@attr.s
class ChangSampleIndicesFromStim(ProcessStep):
    expects = ['stim_diff', 'fs_signal']
    outputs = ['sample_index_map']

    window_size = attr.ib(pd.Timedelta(0.5, 's'))
    #window_size_samples = attr.ib(None)

    label_region_size = attr.ib(pd.Timedelta(1, 's'))
    stim_silence_offset = attr.ib(pd.Timedelta(2, 's'))
    stim_speaking_offset = attr.ib(pd.Timedelta(0.5, 's'))

    def step(self, data_map):
        return self.make_sample_indices(data_map, win_size=self.window_size, label_region_size=self.label_region_size,
                                        silence_offs=self.stim_silence_offset, speaking_offs=self.stim_speaking_offset)

    @staticmethod
    def make_sample_indices(data_map, win_size, label_region_size, silence_offs, speaking_offs):
        """
        Extract 100 sample windows (i.e. .5 seconds) from speaking and silence regions, use defintions:

        [t , t + 0.5s] = no label
        [t + 0.5s, t+ 1.5s] = speaking
        [t + 1.5s, t +2s] = no label
        [t + 2s, t + 3s ] = silence
        t +3 should be the beginning of the next t stimcode onset
        """
        label_index = data_map['stim_diff']
        fs = data_map['fs_signal']

        max_window_samples = int(fs * win_size.total_seconds())
        label_region_sample_size = int(fs * label_region_size.total_seconds())
        print((fs, win_size))
        print("Max window size: %d" % max_window_samples)

        sample_indices = dict()
        ####
        # Separate into two sets and loops so that the negative
        # samples can be derived from the positive samples
        pos_ix = label_index[label_index > 0]
        #neg_ix = label_index[label_index < 0]
        pos_grp = pos_ix.groupby(pos_ix)
        #neg_grp = neg_ix.groupby(neg_ix)
        #speak_offset = pd.Timedelta(0.5, 's')

        # Positive windows extracted first - this way negative windows
        # can potentially be extracted based on the postive windows
        for wrd_id, wave_wrd_values in pos_grp:
            start_t = wave_wrd_values.index.min()
            silence_start_t = start_t + silence_offs
            speaking_start_t = start_t + speaking_offs

            # Get the window starting indices for each region of interest
            # Note on :-max_window_samples - this removes the last windows worth since windows starting here would have out of label samples
#            silence_start_ixes = (label_index[silence_start_t: silence_start_t + label_region_size]
#                                      .index.tolist()[:-max_window_samples])
#            speaking_start_ixes = (label_index[speaking_start_t: speaking_start_t + label_region_size]
#                                       .index.tolist()[:-max_window_samples])
            silence_start_ixes = (label_index.loc[silence_start_t:].iloc[:label_region_sample_size]
                                      .index.tolist()[:-max_window_samples])
            speaking_start_ixes = (label_index[speaking_start_t:].iloc[:label_region_sample_size]
                                       .index.tolist()[:-max_window_samples])


            # Go through the labeled region indices and pull a window of data
            silence_indices = [label_index.loc[offs:offs+win_size].iloc[:max_window_samples].index
                               for offs in silence_start_ixes]
            sample_indices[-wrd_id] = silence_indices

            speaking_indices = [label_index.loc[offs:offs + win_size].iloc[:max_window_samples].index
                                for offs in speaking_start_ixes]
            sample_indices[wrd_id] = speaking_indices


        for w, ixes in sample_indices.items():
            if len(ixes) == 0:
                msg = ("Error: word %d had no windows" % w)
                print(msg)

            bad_win_ixes = [ix for ix in ixes if len(ix) != max_window_samples]
            if len(bad_win_ixes) > 0:
                msg = ("WARNING: word %d had %d windows with wrong sizes" % (w, len(bad_win_ixes)))
                print(msg)

            #for ix in ixes:
            #    if len(ix) != max_window_samples:
            #        msg = ("Warnging: word %d had window with len %d" % (w, len(ix)))
            #        print(msg)

        # Hacky, but just remove anything that's not the right side from the end
        sample_indices = {w: [ix for ix in ixes if len(ix) == max_window_samples]
                          for w, ixes in sample_indices.items()}
        # TODO: Compute flat_index_map and flat_keys and return
        return dict(sample_index_map=sample_indices)

@attr.s
class ChangECOGEnvelope(ProcessStep):
    expects = ['ecog']
    outputs = ['chang_envelope']

    sfreq = attr.ib(1200)

    def step(self, data_map):
        return dict(chang_envelope=self.chang_envelope(data_map['ecog'], sfreq=self.sfreq))


    @staticmethod
    def chang_envelope(ecog_df, band=(70, 150), sfreq=1200):
        #ecog_df = data_map['ecog']
        filt_df = ecog_df.apply(lambda s: filter(s, band, sfreq=sfreq))
        chang_df = filt_df.apply(lambda s: make_hilbert_df(s)['envelope'].rename(f'{s.name}_chang_env'))
        # Resample to 200 times a second (200HZ)
        # TODO: Is taking first correct? Or Some ofther method?
        chang_df = chang_df.resample(str(1 / 200) + 'S').first()
        return chang_df



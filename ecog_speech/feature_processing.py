import numpy as np
import pandas as pd
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

###
@attr.s
class ProcessPipeline:
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
        if not isinstance(other, ProcessStep):
            raise ValueError("Online ProcessStep types allowed for now :|")

        self.steps.append((other, dict()))
        return self


@attr.s
class ProcessStep:
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
            for k, v in  self.input_remap.items():
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
            for k, v in self.output_remap.items():
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
        return self.default_preprocessing(data_map, self.verbose)

    @staticmethod
    def default_preprocessing(data_map, verbose=False):
        stim_s = data_map['stim']
        stim_diff_s = data_map['stim_diff']
        ######
        # Stim event codes and txt
        # 0 is neutral, so running difference will identify the onsets
        # stim_diff_s = stim_s.diff().fillna(0).astype(int)
        # return dict(stim_diff=stim_diff_s)

        word_code_d = data_map['word_code_d']
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
    name = 'threshold'

    #input_override = attr.ib(None)
    threshold = attr.ib(0.007)

    #input_remap = attr.ib(None)#attr.Factory(lambda :dict(stim='stim_pwrt', stim_diff='stim_pwrt_diff')))
    output_remap = attr.ib(attr.Factory(lambda :dict(stim='stim_pwrt', stim_diff='stim_pwrt_diff')))

    expects = ['audio', 'stim']
    outputs = ['stim_pwrt', 'stim_pwrt_diff', 'rolling_audio_pwr']

    def step(self, data_map):
        return self.stim_adjust__power_threshold(data_map, threshold=self.threshold)

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
        #remaps = dict(stim='stim_pwrt', stim_diff='stim_pwrt_diff')
        #return updates, remaps
        return updates

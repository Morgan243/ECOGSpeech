import numpy as np
import pandas as pd


def peak_normalization(arr_s, output_type=int):
    """
    Divide by max value present in array, multiply by the datatypes max, convert to integer
    :param arr_s:
    :return:
    """
    return ((arr_s / arr_s.max()) * np.iinfo(arr_s.dtype).max).round().astype(output_type)


def compute_speaking_mask(wav, power_win_size=1024, avg_power_thresh=50, avg_power_win_size=4096, rate=48000):
    pwr_s = wav.abs().rolling(power_win_size).mean()
    return (pwr_s > avg_power_thresh).rolling(avg_power_win_size).max() > 0


def speech_region_index_from_speech_mask(spk_mask_s):
    leading_edge_s = ((spk_mask_s.diff().replace(0, np.nan) + 1) / 2)
    word_trail_label_s = leading_edge_s.fillna(0).cumsum()
    return spk_mask_s * word_trail_label_s


def compute_speech_index(wav, speaking_mask_f=compute_speaking_mask):
    """
    Provide a function that
    """
    spk_mask_s = speaking_mask_f(wav).astype(int)
    return speech_region_index_from_speech_mask(spk_mask_s)

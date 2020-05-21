import numpy as np
import pandas as pd


def peak_normalization(arr_s, output_type=int):
    """
    Divide by max value present in array, multiply by the datatypes max, convert to integer
    :param arr_s:
    :return:
    """
    return ((arr_s / arr_s.max()) * np.iinfo(arr_s.dtype).max).round().astype(output_type)

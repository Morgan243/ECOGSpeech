import pandas as pd
import numpy as np

from BCI2kReader import BCI2kReader as b2k
from os import environ
import os
import attr
import socket

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')


@attr.s
class NorthwesternWords:

    env_key = 'NORTHWESTERNWORDS_DATASET'
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 pkg_data_dir))
    #attr.ib()
    sensor_columns = list(range(64))
    audio_sample_rate = 48000

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
    def load_dat_file(cls, patient=22, session=1, recording=1, base_path=None, as_frame=True):
        fname = cls.make_filename(patient, session, recording) + ".dat"
        base_path = cls.default_base_path if base_path is None else base_path
        p = os.path.join(base_path, fname)
        r = b2k.BCI2kReader(p)

        if as_frame:
            df = pd.DataFrame(r.signals.T)
            df.index = pd.TimedeltaIndex(df.index / 1200, unit='s')
            
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

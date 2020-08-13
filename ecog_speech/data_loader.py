import pandas as pd
import numpy as np

from BCI2kReader import BCI2kReader as b2k
from os import environ
import os
import attr
import torchaudio
import socket
from ecog_speech import feature_processing

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')

class HarvardSentences:
    pass

import torch
from torch.utils import data as tdata

from tqdm.auto import tqdm

class BaseDataset(tdata.Dataset):
    def to_dataloader(self, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      **kwargs):
        dset = self
        if random_sample:
            if batches_per_epoch is None:
                batches_per_epoch = len(dset)//batch_size

            dataloader = tdata.DataLoader(dset, batch_size=batch_size,
                                         sampler=tdata.RandomSampler(dset,
                                                                    replacement=True,
                                                                    num_samples=batches_per_epoch * batch_size),
                                         shuffle=False, num_workers=num_workers,
                                         **kwargs)
        else:
            dataloader = tdata.DataLoader(dset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers,
                                         **kwargs)
        return dataloader

@attr.s
class NorthwesternWords(BaseDataset):

    env_key = 'NORTHWESTERNWORDS_DATASET'
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 pkg_data_dir))
    #num_mfcc = 13
    sensor_columns = list(range(64))
    audio_sample_rate = 48000
    ecog_sample_rate = 1200

    # In terms of audio samples
    fixed_window_size = attr.ib(audio_sample_rate*1)
    fixed_window_step = attr.ib(int(audio_sample_rate*.01))
    max_ecog_window_size = attr.ib(600)
    num_mfcc = attr.ib(13)

    # Trying this out: define a small data schema class
    schema_fields = ["ecog", "ecog_arr",
                     "speech", "speech_arr",
                     #"mfcc",
                     "mfcc_arr",
                     "text", "text_arr"]
    sample_schema = attr.make_class("NorthwesterWordsSample",
                                    schema_fields)
    sample_schema2 = attr.make_class("NorthwesterWordsSample2",
                                     {f: attr.ib(None) for f in schema_fields})

    def __attrs_post_init__(self):
        self.ecog_df = self.load_dat_to_frame()
        self.speech_df = self.load_wav_to_frame()

        # Important piece - identifies time segments automatically
        #   - Positive values are word/speech labels
        #   - Negative values are silence before the speech (i.e. -1 before 1
        self.word_index = feature_processing.compute_speech_index(self.speech_df)

        self.mfcc_m = torchaudio.transforms.MFCC(self.audio_sample_rate,
                                                 self.num_mfcc)

        #
        self.sample_indices_map = self.make_sample_indices(self.word_index,
                                                           win_size=self.fixed_window_size,
                                                           win_step=self.fixed_window_step)
        self.i_to_ix_key = {i:k for i, k in enumerate(self.sample_indices_map.keys())}
        self.sample_map = dict()

    def __len__(self):
        return len(self.sample_indices_map)

    def __getitem__(self, item):
        ix_k = self.i_to_ix_key[item]

        if ix_k not in self.sample_map:
            self.sample_map[ix_k] = self.make_sample_object(self.sample_indices_map[ix_k],
                                                            ix_k[0], self.ecog_df[self.sensor_columns],
                                                            self.speech_df, max_ecog_samples=self.max_ecog_window_size,
                                                            mfcc_f=self.mfcc_m)
        so = self.sample_map[ix_k]

        return {k: v for k, v in attr.asdict(so).items()
                if isinstance(v, torch.Tensor)}

    def sample_plot(self, i, band=None, figsize=(15, 10)):
        import matplotlib
        t_word_ix = self.word_index[self.word_index == i].index
        t_word_ecog_df = self.ecog_df.reindex(t_word_ix).dropna()
        t_word_wav_df = self.speech_df.reindex(t_word_ix)
        scols = self.sensor_columns

        ecog_std = self.ecog_df[scols].std()
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=ecog_std.min(), vmax=ecog_std.max())
        c_f = lambda v: cmap(norm(v))
        colors = ecog_std.map(c_f).values

        fig, axs = matplotlib.pyplot.subplots(figsize=figsize, nrows=2)

        if band is not None:
            plt_df = t_word_ecog_df[scols].pipe(feature_processing.filter, band=band,
                                                sfreq=self.ecog_sample_rate)
        else:
            plt_df = t_word_ecog_df[scols]

        ax = plt_df.plot(alpha=0.3, legend=False,
                         color=colors, lw=1.2,
                         ax=axs[0], fontsize=14)

        ax = t_word_wav_df.plot(alpha=0.7, legend=False, fontsize=14, ax=axs[1])

        fig.tight_layout()

        return ax

    @classmethod
    def make_sample_indices(cls, label_index, win_size, win_step):
        sample_indices = dict()
        grp = label_index.groupby(label_index)
        for wrd_id, wave_wrd_values in grp:
            ixes = wave_wrd_values.index
            if win_size > len(ixes):
                continue

            for _ix in range(0, len(ixes) - win_size, win_step):
                _ixes = ixes[_ix:_ix + win_size]
                sample_indices[(wrd_id, _ix)] = _ixes

        return sample_indices

    @classmethod
    def make_sample_object(cls, ix, text_label, ecog_df,
                           speech_df, max_ecog_samples,
                           mfcc_f=None):
        mfcc_f = (torchaudio.transforms.MFCC(cls.audio_sample_rate)
                    if mfcc_f is None else mfcc_f)

        text_enc_f = lambda wrd_ix, _text: 0 if wrd_ix == 0 else 1
        kws = dict()
        # TODO: Reindexing may cause problems here - need tolerance param?
        #kws['ecog'] = ecog_df.reindex(ix).dropna()
        kws['ecog'] = ecog_df.loc[ix.min():].dropna().iloc[:max_ecog_samples]
        # Transpose to keep time as last index for torch
        kws['ecog_arr'] = torch.from_numpy(kws['ecog'].values.T).float()
        # Can't do this for non-contihous label regions
        # ecog=nww.ecog_df.loc[ixes.min(): ixes.max()],
        # Word index is base on wave file, so just reindex
        kws['speech'] = speech_df.reindex(ix)
        kws['speech_arr'] = torch.from_numpy(kws['speech'].values).float()
        # print("-----")
        # print(kws['speech_arr'].shape)
        # print(kws['speech_arr'])
        kws['mfcc_arr'] = mfcc_f(kws['speech_arr'])
        kws['text'] = '<silence>' if text_label == 0 else '<speech>'
        kws['text_arr'] = torch.Tensor({'<silence>': [0], '<speech>': [1]}.get(kws['text']))

        s = cls.sample_schema(**kws)
        return s

    #######
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
    def get_path(cls, extension, patient=22, session=1, recording=1, base_path=None):
        fname = cls.make_filename(patient, session, recording) + extension
        base_path = cls.default_base_path if base_path is None else base_path
        p = os.path.join(base_path, fname)
        return p
    @classmethod
    def get_data_path(cls, patient=22, session=1, recording=1, base_path=None):
        return cls.get_path(extension=".dat", patient=patient, session=session,
                            recording=recording, base_path=base_path)

    @classmethod
    def get_wave_path(cls, patient=22, session=1, recording=1, base_path=None):
        return cls.get_path(extension=".wav", patient=patient, session=session,
                            recording=recording, base_path=base_path)

    @classmethod
    def load_dat_file(cls, patient=22, session=1, recording=1, base_path=None, as_frame=True):
        #fname = cls.make_filename(patient, session, recording) + ".dat"
        #base_path = cls.default_base_path if base_path is None else base_path
        #p = os.path.join(base_path, fname)
        p = cls.get_data_path(patient=patient, session=session, recording=recording, base_path=base_path)
        r = b2k.BCI2kReader(p)

        if as_frame:
            df = pd.DataFrame(r.signals.T)
            df.index = pd.TimedeltaIndex(df.index / cls.ecog_sample_rate, unit='s')
            
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

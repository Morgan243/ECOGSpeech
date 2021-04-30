import attr
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from ecog_speech import utils
import matplotlib

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def copy_model_state(m):
    from collections import OrderedDict
    s = OrderedDict([(k, v.cpu().detach().clone())
                      for k, v in m.state_dict().items()])
    return s

## Modules
class Permute(torch.nn.Module):
    def __init__(self, p):
        super(Permute, self).__init__()
        self.p = p

    def forward(self, input):
        return input.permute(*self.p)#(input.size(0), *self.shape)


class Flatten(torch.nn.Module):
    def __init__(self, shape=(-1,)):
        super(Flatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.reshape(input.shape[0], -1)

class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

class CogAttn(torch.nn.Module):
    def __init__(self, trailing_dim, in_channels=64, pooling_size=50):
        print("Cog attn trailing dim (..., bands, time): " + str(trailing_dim))
        print("Cog attn in_channels: " + str(in_channels))
        super(CogAttn, self).__init__()

        self.sensor_repr_model = torch.nn.Sequential(
            # Aggregate along time dimension (mean) to downsample
            torch.nn.AvgPool2d((1, pooling_size)),
            # Convolve along time dimension - i.e. small temporal filters
            torch.nn.Conv2d(in_channels, in_channels, (1, 3)),
            # flatten out bands and time dimensions, leaving in channels intact
            Reshape((in_channels, -1)),
        )
        t_x = torch.rand(16, in_channels, *trailing_dim )
        t_s_out = self.sensor_repr_model(t_x)
        self.attn_trf = torch.nn.Sequential(torch.nn.Linear(t_s_out.shape[-1], in_channels),
                                                torch.nn.ReLU())
#        self.attn_sensor_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(t_s_out.shape[-1], in_channels),
#                                                  torch.nn.ReLU())
#                              for ch in range(in_channels)])
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        repr_out = self.sensor_repr_model(x)
        #attn_sensor_out = torch.cat([self.attn_sensor_layers[ch](repr_out.select(1, ch)).unsqueeze(1)
        #                             for ch in range(repr_out.shape[1])], 1)
        # TODO: Do this as full width convolution, no padding?
        attn_sensor_out = torch.cat([self.attn_trf(repr_out.select(1, ch)).unsqueeze(1)
                                     for ch in range(repr_out.shape[1])], 1)

        attn_softmax = self.softmax(attn_sensor_out)
        attended_out_l = [(attn_softmax[:, ch, :].unsqueeze(1).unsqueeze(1)
                           * x.permute(0, 2, 3, 1)).sum(-1, keepdim=True)
                          for ch in range(attn_softmax.shape[1])
                          ]
        y = torch.cat(attended_out_l, -1).permute(0, 3, 1, 2)
        return y
        # attended_out.shape

from ecog_speech.models import kaldi_nn
SincNN = kaldi_nn.SincConv

class MultiChannelSincNN(torch.nn.Module):
    def __init__(self, num_bands, num_channels,
                 kernel_size=701, fs=200,
                 min_low_hz=1, min_band_hz=3,
                 update=True, per_channel_filter=False,
                 channel_dim=1, unsqueeze_dim=1,
                 padding=0,
                 #store_param_history=False
                 ):
        super(MultiChannelSincNN, self).__init__()
        self.channel_dim = channel_dim
        self.unsqueeze_dim = unsqueeze_dim
        #self.store_param_history = store_param_history
        sinc_kwargs = dict(in_channels=1, out_channels=num_bands,
                           kernel_size=kernel_size, sample_rate=fs,
                           min_low_hz=min_low_hz, min_band_hz=min_band_hz,
                           padding=padding,
                           #update=update,
                           )
        if per_channel_filter:
            # Each channel get's it's own set of filters
            self.sinc_nn_list = [SincNN(**sinc_kwargs)
                                 for c in range(num_channels)]
            self.sinc_nn_list = torch.nn.ModuleList(self.sinc_nn_list)

        else:
            # Each channel uses the same filtering net
            self.sinc_nn = SincNN(**sinc_kwargs)
            self.sinc_nn_list = [self.sinc_nn] * num_channels

        self.set_update(update)

    def set_update(self, update_params=None):
        if update_params is not None:
            self.update_params = update_params

        if self.update_params is not None:
            for n in self.sinc_nn_list:
                n.band_hz_.requires_grad = self.update_params
                n.low_hz_.requires_grad = self.update_params

    def forward(self, x):
        o = [blk(x.select(self.channel_dim, i)).unsqueeze(self.unsqueeze_dim)
             for i, blk in enumerate(self.sinc_nn_list)]
        o = torch.cat(o, self.unsqueeze_dim)
        return o

    def get_band_params(self, trainer=None, to_numpy=True):
        params = [dict(band_hz=snn.band_hz_,
                         low_hz=snn.low_hz_)
                    for snn in self.sinc_nn_list]
        if to_numpy:
            params = [{k:v.clone().cpu().detach().numpy() for k, v in p.items()}
                      for p in params]
        return params

class BaseCNN(torch.nn.Module):
    def __init__(self, in_channels, window_size,
                 dropout=0., in_channel_dropout_rate=0.,
                 dropout2d=False,
                 batch_norm=False,
                 dense_width=None,
                 n_cnn_filters=None
                 #dense_depth=1
                 ):

        super().__init__()
        self.dropout = dropout
        self.activation_cls = torch.nn.SELU
        #DrpOut = torch.nn.Dropout2d if dropout2d else torch.nn.Dropout
        self.dropout_cls = torch.nn.Dropout2d if dropout2d else torch.nn.AlphaDropout
        self.n_cnn_filters = 32 if n_cnn_filters is None else n_cnn_filters

        def make_block(in_ch, out_ch, k_s, s, d, g):
            b = []
            if dropout > 0:
                b.append(self.dropout_cls(self.dropout))
            b.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=k_s, stride=s,
                                     dilation=d, groups=g))
            if batch_norm:
                b.append(torch.nn.BatchNorm2d(out_ch))
            b.append(self.activation_cls())
            return b

        layer_list = [Unsqueeze(1)]

        if in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(in_channel_dropout_rate))

        layer_list += [
            *make_block(1, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=1, g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(3, 3), s=(1, 1), d=1, g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(3, 3), s=(1, 1), d=1, g=1),
            # *make_block(64, 64, k_s=(3, 3), s=(1, 1), d=1, g=1),
            Flatten(),
            self.dropout_cls(self.dropout)
        ]

        self.m = torch.nn.Sequential(*layer_list )
        t_in = torch.rand(32, in_channels, window_size)
        print("T input shape: " + str(t_in.shape))
        t_out = self.m(t_in)
        print("T output shape: " + str(t_out.shape))
        self.dense_width = dense_width
        if self.dense_width is not None:
            self.m.add_module("lin_h0", torch.nn.Linear(t_out.shape[-1], self.dense_width))
            self.m.add_module('act_h0', self.activation_cls())
            self.m.add_module("drp_h0", torch.nn.Dropout(self.dropout))
            self.m.add_module("lin_output", torch.nn.Linear(self.dense_width, 1))
        else:
            self.m.add_module("lin_output", torch.nn.Linear(t_out.shape[-1], 1))

        self.m.add_module('sigmoid_output', torch.nn.Sigmoid())
        self.n_params = utils.number_of_model_params(self.m)
        utils.print_sequential_arch(self.m, t_in)
        print("N params: " + str(self.n_params))

    def forward(self, x):
        return self.m(x)

class BaseMultiSincNN(torch.nn.Module):
    def __init__(self, in_channels, window_size, fs,
                 n_bands=2, per_channel_filter=False,
                 sn_kernel_size=31,
                 sn_padding=0,
                 dropout=0.,
                 dropout2d=False,
                 batch_norm=False,
                 n_cnn_filters=None,
                 dense_width=None,
                 cog_attn=False,
                 in_channel_dropout_rate=0.,
                 make_block_override=None,
                 activation_cls=None,
                 #dense_depth=1
                 ):

        super().__init__()
        self.fs = fs
        self.in_channels = in_channels
        self.window_size = window_size
        self.n_bands = n_bands
        self.sn_kernel_size = sn_kernel_size
        self.sn_padding = sn_padding
        self.batch_norm = batch_norm
        self.cog_attn = cog_attn
        self.in_channel_dropout_rate = in_channel_dropout_rate
        self.make_block_override = make_block_override
        self.dropout = dropout
        #self.activation_cls = torch.nn.SELU if activation_cls is None else activation_cls
        self.activation_cls = torch.nn.PReLU if activation_cls is None else activation_cls
        self.per_channel_filter = per_channel_filter
        #DrpOut = torch.nn.Dropout2d if dropout2d else torch.nn.Dropout
        #self.dropout_cls = torch.nn.Dropout2d if dropout2d else torch.nn.AlphaDropout
        self.dropout_cls = torch.nn.Dropout2d if dropout2d else torch.nn.Dropout
        self.n_cnn_filters = 32 if n_cnn_filters is None else n_cnn_filters

        self.t_in = t_in = torch.rand(32, self.in_channels, self.window_size)
        self.layer_list = self.make_cnn_layer_list()

        self.m = torch.nn.Sequential(*self.layer_list)

        t_out = self.m(t_in)
        print(t_out.shape)
        self.dense_width = dense_width
        if self.dense_width is not None:
            self.m.add_module("lin_h0", torch.nn.Linear(t_out.shape[-1], self.dense_width))
            self.m.add_module('act_h0', self.activation_cls())
            self.m.add_module("drp_h0", torch.nn.Dropout(self.dropout))
            self.m.add_module("lin_output", torch.nn.Linear(self.dense_width, 1))
        else:
            self.m.add_module("lin_output", torch.nn.Linear(t_out.shape[-1], 1))

        self.m.add_module('sigmoid_output', torch.nn.Sigmoid())
        self.n_params = utils.number_of_model_params(self.m)
        utils.print_sequential_arch(self.m, t_in)
        print("N params: " + str(self.n_params))

    def make_cnn_layer_list(self):
        def make_block(in_ch, out_ch, k_s, s, d, g):
            b = []
            if self.dropout > 0:
                b.append(self.dropout_cls(self.dropout))
            b.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=k_s, stride=s,
                                     dilation=d, groups=g))
            if self.batch_norm:
                b.append(torch.nn.BatchNorm2d(out_ch))
            b.append(self.activation_cls())
            return b
        if self.make_block_override is not None:
            make_block = self.make_block_override

        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter),
        )

        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        layer_list += make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1)
        layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1),
        layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1),
        layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1),
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list

    def forward(self, x):
        return self.m(x)

    def get_band_params(self, trainer=None, get_band_kws=None):
        get_band_kws = dict() if get_band_kws is None else get_band_kws
        parameters = list()
        for i, _m in enumerate(self.m):
            is_m = isinstance(_m,
                              MultiChannelSincNN)
            if is_m:
                # in case there are multiple...?
                parameters.append(_m.get_band_params(trainer, **get_band_kws))

        if len(parameters) == 1:
            return parameters[0]
        else:
            return parameters


    @staticmethod
    def parse_band_parameter_training_hist(batch_results, fs=1200, min_low_hz=1, min_band_hz=3):
        channel_param_bandhz_map = dict()
        channel_param_lowhz_map = dict()

        for bi, batch_list in enumerate(batch_results):
            for ch_i, ch_res_d in enumerate(batch_list):
                if ch_i not in channel_param_bandhz_map:
                    channel_param_bandhz_map[ch_i] = list()
                    channel_param_lowhz_map[ch_i] = list()

                channel_param_bandhz_map[ch_i].append(ch_res_d['band_hz'].reshape(-1))
                channel_param_lowhz_map[ch_i].append(ch_res_d['low_hz'].reshape(-1))
        ###*****
        lowhz_df_map = {ch_i: pd.DataFrame(d) * fs + min_low_hz
                        for ch_i, d in channel_param_lowhz_map.items()}

        highhz_df_map = {ch_i: pd.DataFrame(d) * fs + min_band_hz + lowhz_df_map[ch_i]
                         for ch_i, d in channel_param_bandhz_map.items()}

        centerhz_df_map = {ch_i: (lowhz_df_map[ch_i] + highhz_df_map[ch_i]) / 2
                           for ch_i in lowhz_df_map.keys()}
        return lowhz_df_map, highhz_df_map, centerhz_df_map


    @staticmethod
    def plot_sincnet_batch_results(lowhz_df_map, highhz_df_map, centerhz_df_map):
        # TODO: generalize this better
        fig, axs = matplotlib.pyplot.subplots(figsize=(10, 9),
                                              # nrows=len(lowhz_df_map),
                                              nrows=4, ncols=2,
                                              sharex=True)

        ix_slice = slice(None, None, 1000)
        for (ch_i, lowhz_df), (_, highhz_df), (_, centerhz_df) in zip(lowhz_df_map.items(),
                                                                      highhz_df_map.items(),
                                                                      centerhz_df_map.items()):

            try:
                ax = axs.reshape(-1)[ch_i]
            except IndexError as e:
                # dont try to plot more sensors - no more axes!
                break

            for c in lowhz_df.columns:
                centerhz_df.loc[ix_slice][c].plot(ax=ax, lw=3)
                ax.fill_between(centerhz_df.loc[ix_slice].index,
                                lowhz_df.loc[ix_slice][c],
                                highhz_df.loc[ix_slice][c],
                                alpha=0.5)
            ax.grid(True)
            ax.set_ylim(0, 110)
            # ax.set_xlim(0, centerhz_df.shape[0])
            # ax.set_xlim(0, 35*len(dataloader))
            # ax.set_title(ch_i)
            ax.set_ylabel("Channel %d" % ch_i, fontsize=13)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def bandwidth_regularizer(model, w=0.1):
        bp_l = model.get_band_params(get_band_kws=dict(to_numpy=False))
        #bw_arr = np.concatenate([bp_d['band_hz'] for bp_d in bp_l], axis=1).T
        bw_arr = torch.cat([bp_d['band_hz'].unsqueeze(-1) for bp_d in bp_l], -1).T
        if not model.per_channel_filter:
            bw_arr = bw_arr[0]

        return torch.norm(bw_arr) * w


class MultiDim_BNorm1D(torch.nn.Module):
    """Use to norm on the last dim across multiple indices in another dim - basically 1d norm on 2d data"""
    def __init__(self, n_sensors, n_bands, norm_dim=1):
        super(MultiDim_BNorm1D, self).__init__()
        self.norm_dim = norm_dim
        # One temporal normalization per n_norms (n bands)
        self.bnorms = torch.nn.ModuleList([torch.nn.BatchNorm1d(n_bands) for n in range(n_sensors)])

    def forward(self, x):
        o = torch.cat([bn(x.select(self.norm_dim, i)).unsqueeze(self.norm_dim) for i, bn in enumerate(self.bnorms)],
                      dim=self.norm_dim)
        return o
        # return x.squeeze()


class TimeNormBaseMultiSincNN(BaseMultiSincNN):
    def make_cnn_layer_list(self):
        def make_block(in_ch, out_ch, k_s, s, d, g, dropout=self.dropout, batch_norm=self.batch_norm):
            b = []
            if dropout > 0:
                b.append(self.dropout_cls(dropout))
            b.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=k_s, stride=s,
                                     dilation=d, groups=g))
            if batch_norm:
                # b.append(torch.nn.BatchNorm2d(out_ch))
                # b.append(base.Unsqueeze(-2))
                # b.append(base.Reshape(-1, 1, ))
                b.append(MultiDim_BNorm1D(out_ch, self.n_bands))
                # b.append(torch.nn.BatchNorm2d(out_ch))
                # b.append(base.Squeeze())
            b.append(self.activation_cls())
            return b

        if self.make_block_override is not None:
            make_block = self.make_block_override

        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter),
        )

        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        layer_list += make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1)
        layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1)
        layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                 batch_norm=False)
        layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                 batch_norm=False)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


@attr.attrs
class Trainer:
    model_map = attr.ib()
    opt_map = attr.ib()

    train_data_gen = attr.ib()
    #optim_kwargs = attr.ib(dict(weight_decay=0.2, lr=0.001))
    learning_rate = attr.ib(0.001)
    beta1 = attr.ib(0.5)

    criterion = attr.ib(torch.nn.BCELoss())
    cv_data_gen = attr.ib(None)
    epochs_trained = attr.ib(0)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    epoch_cb_history = attr.ib(attr.Factory(list), init=False)
    batch_cb_history = attr.ib(attr.Factory(list), init=False)
    model_regularizer = attr.ib(None)

    default_optim_cls = torch.optim.Adam

    @classmethod
    def set_default_optim(cls, optim):
        cls.default_optim_cls = optim
        return cls

    def __attrs_post_init__(self):
        self.model_map = {k: v.to(self.device) for k, v in self.model_map.items()}
        #self.opt_map = dict()
        for k, m in self.model_map.items():
            m.apply(self.weights_init)

            if k not in self.opt_map:
                if self.default_optim_cls == torch.optim.Adam:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                       lr=self.learning_rate,
                                                             #weight_decay=0.9,
                                                       betas=(self.beta1, 0.999))
                elif self.default_optim_cls == torch.optim.RMSprop:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                             lr=self.learning_rate)

    def get_best_state(self, model_key='model'):
        if getattr(self, 'best_model_state', None) is not None:
            return self.best_model_state
        else:
            return self.copy_model_state(self.model_map[model_key])

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Sinc' in classname:
            pass
        elif 'Conv' in classname or 'Linear' in classname:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    @staticmethod
    def copy_model_state(m):
        from collections import OrderedDict
        s = OrderedDict([(k, v.cpu().detach().clone())
                          for k, v in m.state_dict().items()])
        return s

    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None,
              batch_cb_delta=3):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        #self.epoch_losses = list()
        #:w
        # :w
        # self.train_batch_results = list()
        #self.batch_results = getattr(self, 'epoch_results', dict(epoch=list(), batch=list()))
        #self.epoch_results = getattr(self, 'epoch_results', dict(epoch=list()))
        self.epoch_batch_res_map = dict()
        self.epoch_res_map = dict()

        self.epoch_cb_history += [{k: cb(self) for k, cb in epoch_callbacks.items()}]
        #self.batch_cb_history += [{k: cb(self) for k, cb in batch_callbacks.items()}]
        self.batch_cb_history = {k: list() for k in batch_callbacks.keys()}

        self.n_samples = len(self.train_data_gen)
        train_loss_key = 'loss'

        with tqdm(total=n_epochs,
                  desc='Training epoch',
                  dynamic_ncols=True
                  #ncols='100%'
                  ) as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                epoch_batch_results = dict()
                with tqdm(total=self.n_samples, desc='-loss-', dynamic_ncols=True) as batch_pbar:
                    for i, data in enumerate(self.train_data_gen):
                        update_d = self.train_inner_step(epoch, data)

                        #self.batch_results['epoch'].append(epoch)
                        #self.batch_results['batch'].append(i)

                        prog_msgs = list()
                        for k, v in update_d.items():
                            # TODO: What about spruious results? Maybe do list of dicts instead?
                            #if k not in self.batch_results:
                            if k not in epoch_batch_results:
                                #self.batch_results[k] = [v]
                                epoch_batch_results[k] = [v]
                            else:
                                #self.batch_results[k].append(v)
                                epoch_batch_results[k].append(v)


                            v_l = np.mean(epoch_batch_results[k])
                            prog_msgs.append(f"{k}: {np.round(v_l, 4)}")


                        msg = " || ".join(prog_msgs)
                        # Save Losses for plotting later
                        #G_losses.append(errG.item())
                        #D_losses.append(errD.item())
                        #batch_pbar.set_description("Gen-L: %.3f || Disc-L:%.3f" % (np.mean(G_losses[-20:]),
                        #                                                           np.mean(D_losses[-20:])))
                        batch_pbar.set_description(msg)
                        batch_pbar.update(1)
                        for k, cb in batch_callbacks.items():
                            self.batch_cb_history[k].append(cb(self))

                        #if not i % batch_cb_delta:
                        #    self.batch_cb_history.append({k: cb(self) for k, cb in batch_callbacks.items()})

                #self.epoch_losses.append(dict(gen_losses=G_losses, disc_losses=D_losses))
                #self.epoch_losses.append(epoch_results)
                #self.train_batch_results.append(epoch_results)
                self.epoch_batch_res_map[epoch] = epoch_batch_results
                self.epoch_res_map[epoch] = {k: np.mean(v) for k, v in epoch_batch_results.items()}

                self.epochs_trained += 1
                self.epoch_cb_history.append({k: cb(self) for k, cb in epoch_callbacks.items()})
                if self.cv_data_gen:
                    cv_losses = self.eval(epoch, self.cv_data_gen)
                    self.epoch_res_map[epoch]['cv_loss'] = np.mean(cv_losses)

                epoch_pbar.update(1)
        #return self.train_batch_results
        return self.epoch_res_map

    def eval(self, epoch_i, dataloader):
        model = self.model_map['model'].eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    preds = model(_x['ecog_arr'].to(self.device))
                    loss_l.append(self.criterion(preds, _x['text_arr']
                                                 .to(self.device))
                                  .detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                pbar.set_description("Mean Eval Loss: %.5f" % mean_loss)

                if mean_loss < self.best_cv:
                    self.best_model_state = copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    desc = "CV Loss: %.4f [[NEW BEST]]" % mean_loss
                    pbar.set_description(desc)
                    self.best_cv = mean_loss

        self.model_map['model'].train()
        return loss_l

        #return dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
        #            actuals=torch.cat(actuals_l).detach().cpu().int().numpy())

    def train_inner_step(self, epoch_i, data_batch):
        #real_label = 1
        #fake_label = 0
        res_d = dict()

        model = self.model_map['model']
        #gen_model = self.model_map['gen']
        optim = self.opt_map['model']
        #gen_optim = self.opt_map['gen']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        ecog_arr = data_batch['ecog_arr'].to(self.device)
        actuals = data_batch['text_arr'].to(self.device)
        m_output = model(ecog_arr)

        loss = self.criterion(m_output, actuals)

        if self.model_regularizer is not None:
            reg_l = self.model_regularizer(model)
            loss += reg_l
            res_d['bwreg'] = reg_l.detach().cpu().item()

        # print("backward")
        loss.backward()
        optim.step()
        l = loss.detach().cpu().item()
        res_d['loss'] = l
        model = model.eval()
        return res_d

    @classmethod
    def eval_model_on_dataloader(cls, model, dl, device):
        model_in_training = model.training
        preds_l, actuals_l = list(), list()
        with torch.no_grad():
            model.eval()
            for _x in tqdm(dl, desc="Eval on %s" % dl.__class__.__name__):
                preds_l.append(model(_x['ecog_arr'].to(device)))
                actuals_l.append(_x['text_arr'])

        # Leave the model state in whatever it came in as
        if model_in_training:
            model.train()

    # TODO: This requires and instance of the trainer, which is unecessary for evaluation
    #       Consider separating this into a more simple static/class method that accepts
    #       a model and dataloader (or dataset? dataloader adds batching and other aspects...)
    # - First pass at this in eval_model_on_dataloader
    def generate_outputs(self, model_key='model', **dl_map):
        #self.model.eval()
        model = self.model_map[model_key].eval()
        output_map = dict()
        with torch.no_grad():
            for dname, dset in dl_map.items():
                preds_l, actuals_l = list(), list()
                for _x in tqdm(dset, desc="Eval on %s" % dname):
                    preds_l.append(model(_x['ecog_arr'].to(self.device)))
                    actuals_l.append(_x['text_arr'])

                output_map[dname] = dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
                                         actuals=torch.cat(actuals_l).detach().cpu().int().numpy())
        return output_map


import attr
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from ecog_speech import utils
import matplotlib
from ecog_speech.models import kaldi_nn

SincNN = kaldi_nn.SincConv


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


class ScaleByConstant(torch.nn.Module):
    def __init__(self, divide_by):
        super(ScaleByConstant, self).__init__()
        self.divide_by = divide_by

    def forward(self, input):
        return input / self.divide_by


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


class MultiChannelSincNN(torch.nn.Module):
    def __init__(self, num_bands, num_channels,
                 kernel_size=701, fs=200,
                 min_low_hz=1, min_band_hz=3,
                 update=True, per_channel_filter=False,
                 channel_dim=1, unsqueeze_dim=1,
                 padding=0, band_spacing="linear",
                 #store_param_history=False
                 ):
        super(MultiChannelSincNN, self).__init__()
        self.channel_dim = channel_dim
        self.unsqueeze_dim = unsqueeze_dim
        #self.store_param_history = store_param_history
        sinc_kwargs = dict(in_channels=1, out_channels=num_bands,
                           kernel_size=kernel_size, sample_rate=fs,
                           min_low_hz=min_low_hz, min_band_hz=min_band_hz,
                           padding=padding, band_spacing=band_spacing,
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
            # TODO: Are there issues making this a module list? Duplicate modules?
            self.sinc_nn_list = torch.nn.ModuleList([self.sinc_nn] * num_channels)

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
        params = [dict(band_hz=torch.abs(snn.band_hz_) + snn.min_band_hz / snn.sample_rate,
                       low_hz=torch.abs(snn.low_hz_) + snn.min_low_hz / snn.sample_rate)
                    for snn in self.sinc_nn_list]
        if to_numpy:
            params = [{k:v.clone().cpu().detach().numpy() for k, v in p.items()}
                      for p in params]
        return params


class BaseCNN(torch.nn.Module):
    default_activatino_cls = torch.nn.PReLU
    def __init__(self, in_channels, window_size,
                 dropout=0., in_channel_dropout_rate=0.,
                 dropout2d=False,
                 batch_norm=False,
                 dense_width=None,
                 n_cnn_filters=None,
                 activation_cls=None,
                 print_details=True
                 #dense_depth=1
                 ):

        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.activation_cls = self.default_activatino_cls if activation_cls is None else activation_cls
        if isinstance(self.activation_cls, str):
            self.activation_cls = getattr(torch.nn, self.activation_cls)

        self.dropout_cls = torch.nn.Dropout2d if dropout2d else torch.nn.Dropout
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
        if print_details:
            print("T input shape: " + str(t_in.shape))
        t_out = self.m(t_in)
        if print_details:
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
        if print_details:
            utils.print_sequential_arch(self.m, t_in)
            print("N params: " + str(self.n_params))

    def forward(self, x):
        return self.m(x)


class BaseMultiSincNN(torch.nn.Module):
    default_activation_cls = torch.nn.PReLU
    default_batch_norm_cls = torch.nn.BatchNorm2d
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
                 band_spacing='linear',
                 make_block_override=None,
                 activation_cls=None,
                 print_details=True,
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
        self.band_spacing = band_spacing
        self.make_block_override = make_block_override
        self.dropout = dropout
        #self.activation_cls = torch.nn.SELU if activation_cls is None else activation_cls
        self.activation_cls = self.default_activation_cls if activation_cls is None else activation_cls
        if isinstance(self.activation_cls, str):
            self.activation_cls = getattr(torch.nn, self.activation_cls)

        if print_details:
            print("Using activation " + str(self.activation_cls))

        self.batch_norm_cls = self.default_batch_norm_cls# if batch_norm_cls is None else batch_norm_cls
        self.per_channel_filter = per_channel_filter
        #DrpOut = torch.nn.Dropout2d if dropout2d else torch.nn.Dropout
        #self.dropout_cls = torch.nn.Dropout2d if dropout2d else torch.nn.AlphaDropout
        self.dropout_cls = torch.nn.Dropout2d if dropout2d else torch.nn.Dropout
        # Kind of hacky, but allows quick extension of the class
        self.make_block = self.make_cnn_layer_block if self.make_block_override is None else self.make_block_override
        self.n_cnn_filters = 32 if n_cnn_filters is None else n_cnn_filters

        self.t_in = t_in = torch.rand(32, self.in_channels, self.window_size)
        self.layer_list = self.make_cnn_layer_list()
        print(self.make_cnn_layer_list)

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
        if print_details:
            utils.print_sequential_arch(self.m, t_in)
            print("N params: " + str(self.n_params))

    def make_cnn_layer_block(self, in_ch, out_ch, k_s, s, d, g):
        b = []
        if self.dropout > 0:
            b.append(self.dropout_cls(self.dropout))
        b.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=k_s, stride=s,
                                 dilation=d, groups=g))
        if self.batch_norm:
            b.append(self.batch_norm_cls(out_ch))
        b.append(self.activation_cls())
        return b

    def make_cnn_layer_list(self):
        print("=-=-=-=- Original make cnn layer list =-=-=-=-")
        # Layer 1
        # Add the Band dimension, i.e. Size([64, 51, 300]) becomes Size([64, 51, 1, 300])
        layer_list = [Unsqueeze(2)]

        # Layer +
        # Input channel dropout before anything else
        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        # Layer 2
        # Multi Sinc Net - extract bands
        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        # Layer +
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        # Layer 3 - 6 : Convolutions (ungrouped so all inputs are convolved to all outputs)
        #   - BatchNorm is standard 2d: stats at the batch+sensor level are used to standardize
        #
        # Temporal dim convolution, with stride equal to width and dialation to widen receptive field
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1)
        # Temporal dim convolution, with reduced width
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1)
        # Spectral dim convolution, combining all freqeuncies (bands == kernel size))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1)
        # Temporal dim convolution, small width
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1)

        # Flatten and add final dropout before going to Dens layer
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
        o = torch.cat([
            # apply 1D norm layers
            # - the band dimension should be the channel that stats are
            #   calculated across
            bn(
                # Select channel/sensor
                x.select(self.norm_dim, i)
            # Add back the dim it was selected from to all concatenation
            ).unsqueeze(self.norm_dim)

            for i, bn in enumerate(self.bnorms)],
                      dim=self.norm_dim)
        return o


class MultiDim_Conv1D(torch.nn.Module):
    def __init__(self, n_sensors, in_ch, out_ch, k_s, s, d, g, shared=True, conv_dim=1, ):
        super(MultiDim_Conv1D, self).__init__()
        self.conv_dim = conv_dim
        # One temporal normalization per n_norms (n bands)
        ll = [torch.nn.Conv1d(in_ch, out_ch, kernel_size=k_s, stride=s, dilation=d, groups=g)
              for n in range(n_sensors)] if not shared else [torch.nn.Conv1d(in_ch, out_ch, kernel_size=k_s,
                                                                             stride=s, dilation=d, groups=g)] * n_sensors
        self.cnns = torch.nn.ModuleList(ll)

    def forward(self, x):
        o = torch.cat([
            # apply 1D norm layers
            # - the band dimension should be the channel that stats are
            #   calculated across
            cnn(
                # Select channel/sensor
                x.select(self.conv_dim, i)
            # Add back the dim it was selected from to all concatenation
            ).unsqueeze(self.conv_dim)

            for i, cnn in enumerate(self.cnns)],
                      dim=self.conv_dim)
        return o


with_logger = utils.with_logger(prefix_name=__name__)

@with_logger
@attr.attrs
class Trainer:
    model_map = attr.ib()
    opt_map = attr.ib()

    train_data_gen = attr.ib()
    #optim_kwargs = attr.ib(dict(weight_decay=0.2, lr=0.001))
    learning_rate = attr.ib(0.001)
    beta1 = attr.ib(0.5)

    criterion = attr.ib(torch.nn.BCELoss())
    extra_criteria = attr.ib(None) # regularizers here

    # How many epochs of not beating best CV loss by threshold before early stopping
    early_stopping_patience = attr.ib(None)
    # how much better the new score has to be (0 means at least equal)
    early_stopping_threshold = attr.ib(0)

    cv_data_gen = attr.ib(None)
    epochs_trained = attr.ib(0)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    weights_init_f = attr.ib(None)

    epoch_cb_history = attr.ib(attr.Factory(list), init=False)
    batch_cb_history = attr.ib(attr.Factory(list), init=False)
    model_regularizer = attr.ib(None)

    lr_adjust_on_cv_loss = attr.ib(False)
    lr_adjust_on_plateau_kws = attr.ib(None)
    #lr_adjust_metric = attr.ib('')
    model_name_to_lr_adjust = attr.ib(None)

    default_optim_cls = torch.optim.Adam

    @classmethod
    def set_default_optim(cls, optim):
        cls.default_optim_cls = optim
        return cls

    def __attrs_post_init__(self):
        self.model_map = {k: v.to(self.device) for k, v in self.model_map.items()}
        self.scheduler_map = dict()
        #self.opt_map = dict()
        for k, m in self.model_map.items():
            if self.weights_init_f is not None:
                m.apply(self.weights_init_f)

            if k not in self.opt_map:
                if self.default_optim_cls == torch.optim.Adam:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                       lr=self.learning_rate,
                                                             #weight_decay=0.9,
                                                       betas=(self.beta1, 0.999))
                elif self.default_optim_cls == torch.optim.RMSprop:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                             lr=self.learning_rate)

            # Turn on LR scheduler and (no specific model to schedule or this is a specific model to adjust)
            if self.lr_adjust_on_plateau_kws and (self.model_name_to_lr_adjust is None
                                                  or k in self.model_name_to_lr_adjust):
                self.scheduler_map[k] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_map[k],
                                                                                   **self.lr_adjust_on_plateau_kws)

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
        elif ('Conv' in classname or 'Linear' in classname) and getattr(m, 'weight', None) is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    @staticmethod
    def copy_model_state(m):
        from collections import OrderedDict
        s = OrderedDict([(k, v.cpu().detach().clone())
                          for k, v in m.state_dict().items()])
        return s

    def train(self, n_epochs,
              epoch_callbacks=None,
              batch_callbacks=None,
              batch_cb_delta=3):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        self.epoch_batch_res_map = dict()
        self.epoch_res_map = dict()

        self.epoch_cb_history += [{k: cb(self) for k, cb in epoch_callbacks.items()}]
        self.batch_cb_history = {k: list() for k in batch_callbacks.keys()}

        self.n_samples = len(self.train_data_gen)

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

                        prog_msgs = list()
                        for k, v in update_d.items():
                            # TODO: What about spruious results - can't quite infer epoch?
                            #  Maybe do list of dicts instead?
                            # if this result key hasn't been seen, init a list of values
                            if k not in epoch_batch_results:
                                epoch_batch_results[k] = [v]
                            # eLse, just append the new value
                            else:
                                epoch_batch_results[k].append(v)
                            # Expecting numerics due to this code - take mean of the metric/loss
                            v_l = np.mean(epoch_batch_results[k])
                            # build up the prog bar description string
                            prog_msgs.append(f"{k[:5]}: {np.round(v_l, 6)}")

                        msg = " || ".join(prog_msgs)
                        batch_pbar.set_description(msg)
                        batch_pbar.update(1)
                        if batch_cb_delta is None or (not i % batch_cb_delta):
                            for k, cb in batch_callbacks.items():
                                self.batch_cb_history[k].append(cb(self))

                self.epoch_batch_res_map[epoch] = epoch_batch_results
                self.epoch_res_map[epoch] = {k: np.mean(v) for k, v in epoch_batch_results.items()}

                self.epochs_trained += 1
                self.epoch_cb_history.append({k: cb(self) for k, cb in epoch_callbacks.items()})
                # Produce eval results if a cv dataloader was given
                if self.cv_data_gen:
                    cv_losses = self._eval(epoch, self.cv_data_gen)
                    cv_l_mean = np.mean(cv_losses)
                    self.epoch_res_map[epoch]['cv_loss'] = cv_l_mean

                    for m_name, m_sched in self.scheduler_map.items():
                        m_sched.step(cv_l_mean)

                    if self.early_stopping_patience is not None:
                        self.last_best_cv_l = getattr(self, 'last_best_cv_l', np.inf)
                        if (self.last_best_cv_l - cv_l_mean) > self.early_stopping_threshold:
                            self.logger.info("-------------------------")
                            self.logger.info("---New best for early stopping---")
                            self.logger.info("-------------------------")
                            self.last_best_epoch = epoch
                            self.last_best_cv_l = cv_l_mean
                        elif (epoch - self.last_best_epoch) > self.early_stopping_patience:
                            self.logger.info("--------EARLY STOPPING----------")
                            self.logger.info(f"{epoch} - {self.last_best_epoch} > {self.early_stopping_patience} :: {cv_l_mean}, {self.last_best_cv_l}")
                            self.logger.info("-------------------------")
                            break
                    #cv_o_map, new_best = self._eval(epoch, self.cv_data_gen)
                    #self.epoch_res_map[epoch]['cv_loss'] = np.mean(cv_o_map['loss'])
                    #if new_best:
                    #    print(f"[[ NEW BEST: {self.best_cv} ]]")

                epoch_pbar.update(1)
        return self.epoch_res_map

    # Reuses trainer.generate_outputs(), but not being used
    def _eval_v2(self, epoch_i, dataloader, model_key='model'):
        output_map = self.generate_outputs(model_key, CV=dataloader)['CV']
        mean_loss = np.mean(output_map['loss'])
        self.best_cv = getattr(self, 'best_cv', np.inf)
        new_best = mean_loss < self.best_cv
        if new_best:
            self.best_model_state = copy_model_state(self.model_map[model_key])
            self.best_model_epoch = epoch_i
            self.best_cv = mean_loss

        return output_map, new_best

    def _eval(self, epoch_i, dataloader, model_key='model'):
        """
        trainer's internal method for evaluating losses,
        snapshotting best models and printing results to screen
        """
        model = self.model_map[model_key].eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    preds = model(_x['ecog_arr'].to(self.device))
                    actuals = _x['text_arr'].to(self.device)
                    loss = self.criterion(preds, actuals)

                    loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))
                else:
                    reg_l = 0.
                overall_loss = (mean_loss + reg_l)
                if overall_loss < self.best_cv:

                    self.best_model_state = copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map['model'].train()
        return loss_l

        #return dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
        #            actuals=torch.cat(actuals_l).detach().cpu().int().numpy())

    def train_inner_step(self, epoch_i, data_batch):
        """
        Core training method - gradient descent - provided the epoch number and a batch of data and
        must return a dictionary of losses.
        """
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

        crit_loss = self.criterion(m_output, actuals)
        res_d['crit_loss'] = crit_loss.detach().cpu().item()

        if self.model_regularizer is not None:
            reg_l = self.model_regularizer(model)
            res_d['bwreg'] = reg_l.detach().cpu().item()
        else:
            reg_l = 0

        loss = crit_loss + reg_l
        res_d['total_loss'] = loss.detach().cpu().item()
        loss.backward()
        optim.step()
        model = model.eval()
        return res_d

    def generate_outputs(self, model_key='model', **dl_map):
        """
        Evaluate a model the trainer has on a dictionary of dataloaders.
        """
        model = self.model_map[model_key].eval()
        return self.generate_outputs_from_model(model, dl_map, criterion=self.criterion, device=self.device,
                                                to_frames=False)

    @classmethod
    def generate_outputs_from_model_inner_step(cls, model, data_batch, criterion=None, device=None):
        _x_in = data_batch['ecog_arr']
        _y = data_batch['text_arr']
        if device is not None:
            _x_in = _x_in.to(device)
            _y = _y.to(device)

        preds = model(_x_in)
        ret = dict(preds=preds, actuals=_x_in['text_arr'])
        if criterion is not None:
            ret['criterion'] = criterion(preds, _y)

        return ret

    @classmethod
    def generate_outputs_from_model(cls, model, dl_map, criterion=None, device=None,
                                    to_frames=True, win_step=None, win_size=None) -> dict:
        """
        Produce predictions and targets for a mapping of dataloaders. B/c the trainer
        must know how to pair predictions and targets to train, this is implemented here.
        If model is in training mode, model is returned in training mode

        model: torch model
        dl_map: dictionary of dataloaders to eval on

        returns:
        output_map[dl_map_key][{"preds", "actuals"}]
        """
        model_in_training = model.training
        if device:
            model.to(device)

        model.eval()
        output_map = dict()
        with torch.no_grad():
            for dname, dl in dl_map.items():
                preds_l, actuals_l, criterion_l = list(), list(), list()
                dset = dl.dataset

                #if hasattr(dset, 'data_maps'):
                #    assert len(dset.data_maps) == 1

                #data_map = next(iter(dset.data_maps.values()))
                res_d = dict()
                for _x in tqdm(dl, desc="Eval on [%s]" % str(dname)):
                    _inner_d = cls.generate_outputs_from_model_inner_step(model, _x)

                    for k, v in _inner_d.items():
                        curr_v = res_d.get(k, list())
                        new_v = (curr_v + v) if isinstance(v, list) else (curr_v + [v])
                        res_d[k] = new_v

#                    _x_in = _x['ecog_arr']
#                    _y = _x['text_arr']
#                    if device:
#                        _x_in = _x_in.to(device)
#                        _y = _y.to(device)
#
#                    preds_l.append(model(_x_in))
#                    actuals_l.append(_x['text_arr'])
#                    if criterion is not None:
#                        criterion_l.append(criterion(preds_l[-1], _y))
                output_map[dname] = {k: torch.cat(v_l).detach().cpu().numpy() for k, v_l in res_d.items()}
                #output_map[dname] = dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
                #                         actuals=torch.cat(actuals_l).detach().cpu().int().numpy())
                                         #loss=torch.cat(criterion_l).detach().cpu().numpy())
                if to_frames:
                    t_ix = None
                    #if win_step is not None and win_size is not None:
                    #    t_ix = data_map['ecog'].iloc[range(win_size, data_map['ecog'].shape[0], win_step)].index
                    out_df = pd.DataFrame({k: v.squeeze() for k, v in output_map[dname].items()}, index=t_ix)
                    output_map[dname] = out_df
                    #output_map = {out_k: pd.DataFrame({k: v.squeeze() for k, v in preds_map.items()}, index=t_ix)
                    #              for out_k, preds_map in output_map.items()}

                if criterion is not None:
                    output_map[dname]['loss'] = torch.Tensor(criterion_l).detach().cpu().numpy()

        if model_in_training:
            model.train()

#        if to_frames:
#            t_ix = None
#            if win_step is not None and win_size is not None and data_map is not None:
#                t_ix = data_map['ecog'].iloc[range(win_size, data_map['ecog'].shape[0], win_step)].index
#
#            output_map = {out_k: pd.DataFrame({k: v.squeeze() for k, v in preds_map.items()}, index=t_ix)
#                          for out_k, preds_map in output_map.items()}

        return output_map

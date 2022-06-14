import numpy as np
import torch

import base
import base as bmp

from ecog_speech.models.base import (BaseMultiSincNN, Unsqueeze,
                                     MultiChannelSincNN, CogAttn,
                                     Flatten, Permute,
                                     ScaleByConstant,
                                     MultiDim_Conv1D, MultiDim_BNorm1D)

from ecog_speech.models import base as bmp
from ecog_speech import datasets
from typing import Optional, Type
from dataclasses import dataclass, field
from ecog_speech.experiments import base as bxp
from ecog_speech.models import base as bmp
from ecog_speech import utils


logger = utils.get_logger(__name__)

###

class TimeNormBaseMultiSincNN(BaseMultiSincNN):
    #default_batch_norm_cls = MultiDim_BNorm1D

    def make_cnn_layer_block(self, in_ch, out_ch, k_s, s, d, g, batch_norm=None):
        b = []
        if self.dropout > 0:
            b.append(self.dropout_cls(self.dropout))
        b.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=k_s, stride=s,
                                 dilation=d, groups=g))
        if batch_norm is not None and self.batch_norm:
            # Override: Multi dim norm needs to know the number of bands
            if isinstance(batch_norm, list):
                b += batch_norm
            else:
                b.append(batch_norm)
        b.append(self.activation_cls())
        return b

    def make_cnn_layer_list(self):
        print("=-=-=-=- Tnorm make cnn layer list =-=-=-=-")
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1,
                                      batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


#####
class MultiDimBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, agg_dims=(0, 3), eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MultiDimBatchNorm2d, self).__init__()
        # num_features, eps, momentum, affine, track_running_stats)
        self.num_features = list(num_features) if isinstance(num_features, (tuple, list)) else [num_features]
        self.agg_dims = list(agg_dims)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = self.bias = None

        ####
        self.num_batches_tracked = None
        self.num_feature_values = np.prod([nf for nf in self.num_features])
        self.running_mean = None
        self.running_var = None

    def forward(self, x: torch.Tensor):
        d = x.device
        if self.running_mean is None:
            self.running_var = self.running_mean = 0
        else:
            self.running_mean = self.running_mean.to(d)
            self.running_var = self.running_var.to(d)

        exp_avg_factor = 0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exp_avg_factor = 1.0 / self.num_batches_tracked
                else:
                    exp_avg_factor = self.momentum

        if self.training:
            var = x.var(self.agg_dims, unbiased=False, keepdim=True)
            mu = x.mean(self.agg_dims, keepdim=True)
            n = x.numel() / self.num_feature_values
            with torch.no_grad():
                self.running_mean = exp_avg_factor * mu + (1 - exp_avg_factor) * self.running_mean
                self.running_var = exp_avg_factor * var * n / (n -1) + (1 - exp_avg_factor) * self.running_var
        else:
            var = self.running_var
            mu = self.running_mean

        _x = (x - mu) / (torch.sqrt(var + self.eps))
        if self.affine:
            self.weight = self.weight.to(d)
            self.bias = self.bias.to(d)
            _x = _x * self.weight[None, :, :, None] + self.bias[None, :, :, None]
        return _x


class TimeNormBaseMultiSincNN_v2(TimeNormBaseMultiSincNN):
    """Add new custom MultiDimBatchNorm2d immediately following MultiSincNet"""

    def make_cnn_layer_list(self):
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1,
                                 #batch_norm=MultiDimBatchNorm2d([self.n_cnn_filters, self.n_bands], [0, 3])
                                 batch_norm=None
                                 )
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                 #batch_norm=MultiDimBatchNorm2d([self.n_cnn_filters, self.n_bands], [0, 3])
                                 batch_norm=None
                                 )
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                 batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                 batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v3(TimeNormBaseMultiSincNN):
    """Like v2 (new match norm after sincNet), but one fewer CNN layers following"""

    def make_cnn_layer_list(self):
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_block(self.in_channels, self.in_channels, k_s=(1, 5), s=(1, 5), d=(1, 2), g=self.in_channels,
                                     batch_norm=None
                                     )
        #layer_list += make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
       #                          batch_norm=None
       #                          )
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                      batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                      batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list

# Activation can be set now
class TimeNormBaseMultiSincNN_v4(TimeNormBaseMultiSincNN):
#    """Same as v3, only use LeakyReLU to use fewer parameters"""
    default_activation_cls = torch.nn.LeakyReLU


class TimeNormBaseMultiSincNN_v5(TimeNormBaseMultiSincNN):
    """
    1D convolutions rather than 2d immediately following sinc net output
    in attempt to keep sensor output independent
    """
    def make_cnn_layer_list(self):
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #batch_norm = MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        if self.dropout > 0:
            layer_list += [self.dropout_cls(self.dropout)]
        layer_list += [MultiDim_Conv1D(self.in_channels, #self.n_bands,
                                      in_ch=self.n_bands, out_ch=self.n_cnn_filters,
                                      k_s=(5), s=(5), d=(2), g=1)]
        # Rescale all bands within before fusion
        layer_list += [MultiDimBatchNorm2d([self.in_channels, self.n_bands], affine=False)]
        #layer_list += torch.nn.BatchNorm2d(self.in_channels, affine=False)
        layer_list += [self.activation_cls()]
        # Early fusion - all inputs are convolved with all kernels to make all outputs
        #layer_list += self.make_block(self.in_channels, self.in_channels, k_s=(1, 5), s=(1, 5), d=(1, 2), g=self.in_channels,
                                      #batch_norm=None)
        #                              batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
                                      #batch_norm=MultiDimBatchNorm2d([self.n_cnn_filters, self.n_bands], affine=False))
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      batch_norm=None)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                      #batch_norm=MultiDimBatchNorm2d([self.n_cnn_filters, self.n_bands], affine=False))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                      batch_norm=None)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                      #batch_norm=MultiDimBatchNorm2d([self.n_cnn_filters, self.n_bands], affine=False))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                      batch_norm=None)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v6(TimeNormBaseMultiSincNN):
    """
    Use custom multidim after sincnet, then 1-1 grouped
    convolutions, use standard 2d in remaining cnns
    """

    def make_cnn_layer_list(self):
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_block(self.in_channels, self.in_channels, k_s=(1, 5), s=(1, 5), d=(1, 2), g=self.in_channels,
                                     batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     )
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                      batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                      batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v7(TimeNormBaseMultiSincNN):
    """
    Like v6 (custom batch norm, 1-1 grouped, 2d batchnrom) but
    now swap sensor and band dimension so we can use regular 2d bnorm
    to normalize to the band dimension (aggregate sensors, time, and batch samples)
    for the first two CNNs, then swap back to (batch, sensor, band, time) before
    final convolutions across bands then time
    """

    def make_cnn_layer_list(self):
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Swap band and sensor dimension
        layer_list.append(Permute((0, 2, 1, 3)))

        layer1_factor = 2
        n1_filters = layer1_factor * self.n_bands
        layer_list += self.make_block(self.n_bands, n1_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=self.n_bands,
                                     batch_norm=torch.nn.BatchNorm2d(n1_filters, affine=False))
        n2_filters = 2 * n1_filters
        layer_list += self.make_block(n1_filters, n2_filters, k_s=(1, 3), s=(1, 3), d=1, g=n1_filters,
                                      batch_norm=torch.nn.BatchNorm2d(n2_filters, affine=False)
                                     #batch_norm=None
                                     )
        # Swap sensor and band dimensions back
        layer_list.append(Permute((0, 2, 1, 3)))
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                      batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 2), d=1, g=1,
                                      batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                      batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v8(TimeNormBaseMultiSincNN):
    """
    Use custom multi dim norm after sinc net AND after subsequent layers
    (previously used regular batch norm 2d on last layers)
    """
    default_activation_cls = torch.nn.Hardtanh

    def make_cnn_layer_list(self):
        layer_list = [Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_block(self.in_channels, self.in_channels, k_s=(1, 5), s=(1, 5), d=(1, 2), g=self.in_channels,
                                      batch_norm=MultiDim_BNorm1D(self.in_channels, self.n_bands))
                                     #batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     #)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v9(TimeNormBaseMultiSincNN):
    """Like v8 but scale inputs by constant"""

    def make_cnn_layer_list(self):
        layer_list = [ScaleByConstant(128), Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_block(self.in_channels, self.in_channels, k_s=(1, 5), s=(1, 5), d=(1, 2), g=self.in_channels,
                                      batch_norm=MultiDim_BNorm1D(self.in_channels, self.n_bands))
                                     #batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     #)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v10(TimeNormBaseMultiSincNN):
    """Like v9, but no longer grouping kernels after sincNet"""

    def make_cnn_layer_list(self):
        layer_list = [ScaleByConstant(128), Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1,
                                      batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
                                     #batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands))
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     #)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += self.make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v11(TimeNormBaseMultiSincNN):
    """
    User regular 2d batch norm at the band level following the
    sinc net layers - much faster and more stable from initial steps
    """

    def make_cnn_layer_list(self):
        print("=-=-=- Make layer list =-=-=-")
        layer_list = [#ScaleByConstant(128),
                      Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        if self.batch_norm:
            layer_list += [
                # Swap band and sensor
                Permute((0, 2, 1, 3)),
                # Aggregate stats on batch, sensor, time (i.e. band-level)
                torch.nn.BatchNorm2d(self.n_bands),
                # Swap band and sensor back
                Permute((0, 2, 1, 3))
            ]
            #layer_list.append(MultiDimBatchNorm2d([self.n_bands], [0, 1, 3], affine=True))
            #layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))
            #layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=True))
            #layer_list.append(MultiDim_BNorm1D(self.in_channels, self.n_bands))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_cnn_layer_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1,
                                      #batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands)
                                      )
#        layer_list += [
#            # Swap band and sensor
#            Permute((0, 2, 1, 3)),
#            # Aggregate stats on batch, sensor, time (i.e. band-level)
#            torch.nn.BatchNorm2d(self.n_bands),
#            # Swap band and sensor back
#            Permute((0, 2, 1, 3))
#        ]
                                     #batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      #batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands)
                                      )
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     #)
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, ))
                                        #batch_norm=None)
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,)
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, ))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v12(TimeNormBaseMultiSincNN):
    """
    User regular 2d batch norm at the band level following the
    sinc net layers - much faster and more stable from initial steps
    """

    def make_cnn_layer_list(self):
        print("=-=-=- Make layer list =-=-=-")
        layer_list = [#ScaleByConstant(128),
                      Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        if self.batch_norm:
            layer_list += [
                # Swap band and sensor
                Permute((0, 2, 1, 3)),
                # Aggregate stats on batch, sensor, time (i.e. band-level)
                torch.nn.BatchNorm2d(self.n_bands),
                # Swap band and sensor back
                Permute((0, 2, 1, 3))
            ]
            #layer_list.append(MultiDimBatchNorm2d([self.n_bands], [0, 1, 3], affine=True))
            #layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))
            #layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=True))
            #layer_list.append(MultiDim_BNorm1D(self.in_channels, self.n_bands))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_cnn_layer_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1,
                                      #batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands)
                                                batch_norm=[# Swap band and sensor
                                                            Permute((0, 2, 1, 3)),
                                                            # Aggregate stats on batch, sensor, time (i.e. band-level)
                                                            torch.nn.BatchNorm2d(self.n_bands),
                                                            # Swap band and sensor back
                                                            Permute((0, 2, 1, 3))]
                                      )
                                     #batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      #batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands)
                                                batch_norm=[  # Swap band and sensor
                                                    Permute((0, 2, 1, 3)),
                                                    # Aggregate stats on batch, sensor, time (i.e. band-level)
                                                    torch.nn.BatchNorm2d(self.n_bands),
                                                    # Swap band and sensor back
                                                    Permute((0, 2, 1, 3))]
                                      )
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     #)
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                                batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, ))
                                        #batch_norm=None)
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                                batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, ))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


class TimeNormBaseMultiSincNN_v13(TimeNormBaseMultiSincNN):
    """
    User regular 2d batch norm at the band level following the
    sinc net layers - much faster and more stable from initial steps
    Include scaling at top level
    """

    def make_cnn_layer_list(self):
        print("=-=-=- Make layer list =-=-=-")
        layer_list = [ScaleByConstant(128),
                      Unsqueeze(2)]

        if self.in_channel_dropout_rate > 0:
            layer_list.append(torch.nn.Dropout2d(self.in_channel_dropout_rate))

        layer_list.append(
            MultiChannelSincNN(self.n_bands, self.in_channels,
                               padding=self.sn_padding,
                               kernel_size=self.sn_kernel_size, fs=self.fs,
                               per_channel_filter=self.per_channel_filter,
                               band_spacing=self.band_spacing),
        )

        if self.batch_norm:
            layer_list += [
                # Swap band and sensor
                Permute((0, 2, 1, 3)),
                # Aggregate stats on batch, sensor, time (i.e. band-level)
                torch.nn.BatchNorm2d(self.n_bands),
                # Swap band and sensor back
                Permute((0, 2, 1, 3))
            ]
            #layer_list.append(MultiDimBatchNorm2d([self.n_bands], [0, 1, 3], affine=True))
            #layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=False))
            #layer_list.append(MultiDimBatchNorm2d([self.in_channels, self.n_bands], [0, 3], affine=True))
            #layer_list.append(MultiDim_BNorm1D(self.in_channels, self.n_bands))

        #print("VERS 3")
        if self.cog_attn:
            tmp_model = torch.nn.Sequential(*layer_list)
            t_out = tmp_model(self.t_in)
            print("!!-Using attentions-!!")
            layer_list.append(CogAttn((t_out.shape[-2], t_out.shape[-1]), self.in_channels))

        #######
        # Sets up for late fusion - keep one filter per channel
        layer_list += self.make_cnn_layer_block(self.in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1,
                                      #batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands)
                                                batch_norm=[# Swap band and sensor
                                                            Permute((0, 2, 1, 3)),
                                                            # Aggregate stats on batch, sensor, time (i.e. band-level)
                                                            torch.nn.BatchNorm2d(self.n_bands),
                                                            # Swap band and sensor back
                                                            Permute((0, 2, 1, 3))]
                                      )
                                     #batch_norm=torch.nn.BatchNorm2d(self.in_channels, affine=False))
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1,
                                      #batch_norm=MultiDim_BNorm1D(self.n_cnn_filters, self.n_bands)
                                                batch_norm=[  # Swap band and sensor
                                                    Permute((0, 2, 1, 3)),
                                                    # Aggregate stats on batch, sensor, time (i.e. band-level)
                                                    torch.nn.BatchNorm2d(self.n_bands),
                                                    # Swap band and sensor back
                                                    Permute((0, 2, 1, 3))]
                                      )
                                      #batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, affine=False)
                                     #batch_norm=None
                                     #)
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(self.n_bands, 1), s=(1, 1), d=1, g=1,
                                                batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, ))
                                        #batch_norm=None)
        layer_list += self.make_cnn_layer_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 1), d=1, g=1,
                                                batch_norm=torch.nn.BatchNorm2d(self.n_cnn_filters, ))
                                        #batch_norm=None)
        layer_list += [Flatten(), self.dropout_cls(self.dropout)]
        return layer_list


sinc_ieeg_v_map = {
    (m_str := m_cls.__name__.split('_')[-1]): m_cls
    for m_cls in [TimeNormBaseMultiSincNN_v2, TimeNormBaseMultiSincNN_v3,
                  TimeNormBaseMultiSincNN_v4, TimeNormBaseMultiSincNN_v5,
                  TimeNormBaseMultiSincNN_v6, TimeNormBaseMultiSincNN_v7,
                  TimeNormBaseMultiSincNN_v8, TimeNormBaseMultiSincNN_v10,
                  TimeNormBaseMultiSincNN_v11, TimeNormBaseMultiSincNN_v12,
                  TimeNormBaseMultiSincNN_v13]
}


def get_model_cls_from_options_str(model_str):
    model = None
    if 'sn-v' in model_str:
        v = model_str.split('-')[-1]
        model = sinc_ieeg_v_map[v]
    elif model_str == 'tnorm-base-sn':
        model = TimeNormBaseMultiSincNN
    elif model_str == 'base-sn':
        model = bmp.BaseMultiSincNN
    else:
        raise ValueError(f"Don't know model '{model_str}'")

    return model


@dataclass
class SincIEEGOptions(bmp.CNNModelOptions):
    model_name: str = 'sinc_ieeg'

    sn_n_bands: int = 1
    sn_kernel_size: int = 31
    sn_padding: int = 15
    sn_band_spacing: str = 'linear'
    track_sinc_params: bool = False
    roll_channels: bool = False
    shuffle_channels: bool = False
    cog_attn: bool = False
    bw_reg_weight: float = 0.

    version: str = 'tnorm-base-sn-v13'

    def make_model_kws(self, dataset: Optional[Type[datasets.BaseDataset]] = None,
                       in_channels=None, window_size=None):

        return dict(in_channels=int(dataset.get_feature_shape()[0]) if in_channels is None else in_channels,
                    window_size=int(dataset.get_feature_shape()[-1]) if window_size is None else window_size,
                    dropout=self.dropout,
                    in_channel_dropout_rate=self.in_channel_dropout_rate,
                    dropout2d=self.dropout_2d,
                    batch_norm=self.batch_norm,
                    dense_width=self.dense_width,
                    n_cnn_filters=self.n_cnn_filters,
                    activation_cls=self.activation_class,
                    print_details=self.print_details,
                    n_bands=self.sn_n_bands,
                    sn_padding=self.sn_padding,
                    fs=dataset.fs_signal,
                    cog_attn=self.cog_attn,
                    band_spacing=self.sn_band_spacing
                    )

    def make_model(self, dataset: Optional[Type[datasets.BaseDataset]] = None,
                   in_channels=None, window_size=None):
        model_kws = self.make_model_kws(dataset, in_channels=in_channels, window_size=window_size)
        model_cls = get_model_cls_from_options_str(self.version)
        return model_cls(**model_kws), model_kws

    def make_model_regularizer_function(self, model):
        reg_f = None
        if self.bw_reg_weight > 0:
            logger.info(f"!!!! Using BW regularizeer W={self.bw_reg_weight} !!!!")
            reg_f = lambda m: model.bandwidth_regularizer(m, w=self.bw_reg_weight)
        return reg_f


def make_model(options: Type[bmp.DNNModelOptions] = None, nww=None, model_name=None, model_kws=None, print_details=True):
    """
    Helper method - Given command-line options and a NorthwesterWords derived dataset, build the model
    specified in the options.
    """
    assert not (nww is None and model_kws is None)
    base_kws = dict()
    if options is not None:
        base_kws.update(dict(
            #window_size=int(nww.sample_ixer.window_size.total_seconds() * nww.fs_signal),
            # time/samples is the last dimension
            window_size=int(nww.get_feature_shape()[-1]),
            dropout=options.dropout,
            dropout2d=options.dropout_2d,
            batch_norm=options.batchnorm,
            dense_width=options.dense_width,
            activation_cls=options.activation_class,
            print_details=print_details
        ))

    model_name = options.model_name if model_name is None else model_name

    if model_name == 'base-sn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             n_bands=options.sn_n_bands,
                             n_cnn_filters=options.n_cnn_filters,
                             sn_padding=options.sn_padding,
                             sn_kernel_size=options.sn_kernel_size,
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             fs=nww.fs_signal,
                             cog_attn=options.cog_attn,
                             **base_kws)
        model = base.BaseMultiSincNN(**model_kws)
    elif model_name == 'tnorm-base-sn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             n_bands=options.sn_n_bands,
                             n_cnn_filters=options.n_cnn_filters,
                             sn_padding=options.sn_padding,
                             sn_kernel_size=options.sn_kernel_size,
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             fs=nww.fs_signal,
                             cog_attn=options.cog_attn,
                             band_spacing=options.sn_band_spacing,
                             **base_kws)
        model = TimeNormBaseMultiSincNN(**model_kws)
    elif 'tnorm-base-sn-v' in model_name:
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             n_bands=options.sn_n_bands,
                             n_cnn_filters=options.n_cnn_filters,
                             sn_padding=options.sn_padding,
                             sn_kernel_size=options.sn_kernel_size,
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             fs=nww.fs_signal,
                             cog_attn=options.cog_attn,
                             band_spacing=options.sn_band_spacing,
                             **base_kws)
        model_cls = get_model_cls_from_options_str(model_name)
        model = model_cls(**model_kws)
    elif model_name == 'base-cnn':
        if model_kws is None:
            model_kws = dict(in_channels=len(nww.selected_columns),
                             in_channel_dropout_rate=options.in_channel_dropout_rate,
                             n_cnn_filters=options.n_cnn_filters,
                             # band_spacing=options.sn_band_spacing,
                             **base_kws)
        model = base.BaseCNN(**model_kws)
    else:
        msg = f"Unknown model name {model_name}"
        raise ValueError(msg)

    return model, model_kws

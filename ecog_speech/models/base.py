import attr
from tqdm.auto import tqdm
import numpy as np
import torch
from ecog_speech import utils

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

    def get_band_params(self, trainer=None):
        return [dict(band_hz=snn.band_hz_.clone().cpu().detach().numpy(),
                     low_hz=snn.low_hz_.clone().cpu().detach().numpy())
                for snn in self.sinc_nn_list]

class BaseCNN(torch.nn.Module):
    def __init__(self, in_channels, window_size,
                 dropout=0.,
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

        self.m = torch.nn.Sequential(
            Unsqueeze(1),

            #MultiChannelSincNN(n_bands, in_channels,
            #                   padding=sn_padding,
            #                   kernel_size=sn_kernel_size, fs=fs,
            #                   per_channel_filter=per_channel_filter),

            *make_block(1, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=1, g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(3, 3), s=(1, 1), d=1, g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(3, 3), s=(1, 1), d=1, g=1),
            #*make_block(64, 64, k_s=(3, 3), s=(1, 1), d=1, g=1),
            Flatten(),
            self.dropout_cls(self.dropout)

        )
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

        self.m = torch.nn.Sequential(
            Unsqueeze(2),

            MultiChannelSincNN(n_bands, in_channels,
                               padding=sn_padding,
                               kernel_size=sn_kernel_size, fs=fs,
                               per_channel_filter=per_channel_filter),

            *make_block(in_channels, self.n_cnn_filters, k_s=(1, 5), s=(1, 5), d=(1, 2), g=1),
            #*make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 5), s=(1, 3), d=(1, 1), g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(n_bands, 1), s=(1, 1), d=1, g=1),
            *make_block(self.n_cnn_filters, self.n_cnn_filters, k_s=(1, 3), s=(1, 3), d=1, g=1),
            Flatten(),
            self.dropout_cls(self.dropout)

        )
        t_in = torch.rand(32, in_channels, window_size)
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

    def forward(self, x):
        return self.m(x)

    def get_band_params(self, trainer=None):
        parameters = list()
        for i, _m in enumerate(self.m):
            is_m = isinstance(_m,
                              MultiChannelSincNN)
            if is_m:
                # in case there are multiple...?
                parameters.append(_m.get_band_params(trainer))

        if len(parameters) == 1:
            return parameters[0]
        else:
            return parameters


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


    default_optim_cls = torch.optim.Adam

    def __attrs_post_init__(self):
        self.model_map = {k: v.to(self.device) for k, v in self.model_map.items()}
        #self.opt_map = dict()
        for k, m in self.model_map.items():
            m.apply(self.weights_init)

            if k not in self.opt_map:
                if self.default_optim_cls == torch.optim.Adam:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                       lr=self.learning_rate,
                                                       betas=(self.beta1, 0.999))
                elif self.default_optim_cls == torch.optim.RMSprop:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                             lr=self.learning_rate)

    def get_best_state(self, model_key='model'):
        if self.best_model_state is not None:
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
        self.train_batch_results = list()
        self.epoch_results = getattr(self, 'epoch_results', dict(epoch=list(), batch=list()))

        self.epoch_cb_history += [{k: cb(self) for k, cb in epoch_callbacks.items()}]
        #self.batch_cb_history += [{k: cb(self) for k, cb in batch_callbacks.items()}]
        self.batch_cb_history = {k: list() for k in batch_callbacks.keys()}

        self.n_samples = len(self.train_data_gen)

        with tqdm(total=n_epochs,
                  desc='Training epoch',
                  dynamic_ncols=True
                  #ncols='100%'
                  ) as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                with tqdm(total=self.n_samples, desc='-loss-', dynamic_ncols=True) as batch_pbar:
                    for i, data in enumerate(self.train_data_gen):
                        update_d = self.train_inner_step(epoch, data)

                        self.epoch_results['epoch'].append(epoch)
                        self.epoch_results['batch'].append(i)

                        prog_msgs = list()
                        for k, v in update_d.items():
                            # TODO: What about spruious results? Maybe do list of dicts instead?
                            if k not in self.epoch_results:
                                self.epoch_results[k] = [v]
                            else:
                                self.epoch_results[k].append(v)


                            v_l = np.round(np.mean(self.epoch_results[k][-20:]), 4)
                            prog_msgs.append(f"{k}: {v_l}")


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
                self.epochs_trained += 1
                self.epoch_cb_history.append({k: cb(self) for k, cb in epoch_callbacks.items()})
                if self.cv_data_gen:
                    cv_losses = self.eval(epoch, self.cv_data_gen)

                epoch_pbar.update(1)
        #return self.train_batch_results
        return self.epoch_results

    def eval(self, epoch_i, dataloader):
        model = self.model_map['model']
        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    preds = model(_x['ecog_arr'].to(self.device))
                    loss_l.append(self.criterion(preds, _x['text_arr']).detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                pbar.set_description("Mean Eval Loss: %.5f" % mean_loss)

                if mean_loss < self.best_cv:
                    self.best_model_state = copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    desc = "CV Loss: %.4f [[NEW BEST]]" % mean_loss
                    pbar.set_description(desc)
                    self.best_cv = mean_loss
        return loss_l

        #return dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
        #            actuals=torch.cat(actuals_l).detach().cpu().int().numpy())


    def train_inner_step(self, epoch_i, data_batch):
        real_label = 1
        fake_label = 0

        model = self.model_map['model']
        #gen_model = self.model_map['gen']
        optim = self.opt_map['model']
        #gen_optim = self.opt_map['gen']

        model.zero_grad()
        optim.zero_grad()

        ecog_arr = data_batch['ecog_arr'].to(self.device)
        actuals = data_batch['text_arr'].to(self.device)
        m_output = model(ecog_arr)

        loss = self.criterion(m_output, actuals)
        # print("backward")
        loss.backward()
        optim.step()
        l = loss.detach().cpu().item()
        return dict(loss=l)

#    def train_old(self, n_epochs, epoch_callbacks=None, batch_callbacks=None,
#              batch_cb_delta=5):
#
#        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
#        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks
#
#        self.model = self.model.to(self.device)
#
#        # self.criterion = torch.nn.CrossEntropyLoss()
#        self.criterion = torch.nn.BCELoss()  # weight=torch.tensor(2))
#        self.optim = torch.optim.Adam(self.model.parameters(), **self.optim_kwargs)
#        self.losses = getattr(self, 'losses', list())
#        self.cv_losses = getattr(self, 'cv_losses', list())
#        best_cv = np.inf
#        train_loss = 0
#        with tqdm(total=n_epochs, desc='Train epoch') as epoch_pbar:
#            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
#                self.model.train()
#                with tqdm(total=len(self.train_data_gen), desc='-loss-') as batch_pbar:
#                    for batch_idx, data_dict in enumerate(self.train_data_gen):
#                        self.model.zero_grad()
#
#                        # print("batch {i}")
#                        ecog_arr = data_dict['ecog_arr'].to(self.device)
#                        # ecog_arr = (ecog_arr/ecog_arr.abs().max(1, keepdim=True).values)
#
#                        if batch_idx == 0:
#                            pass
#                            #print("ECOG SHAPE: " + str(ecog_arr.shape))
#
#                        actuals = data_dict['text_arr'].to(self.device)
#                        # print("running model")
#                        m_output = self.model(ecog_arr)
#
#                        self.optim.zero_grad()
#                        loss = self.criterion(m_output, actuals)
#                        # print("backward")
#                        loss.backward()
#                        self.optim.step()
#                        l = loss.detach().cpu().item()
#
#                        train_loss += l
#
#                        self.losses.append(l)
#                        mu_loss = np.mean(self.losses[-(batch_idx + 1):])
#                        batch_pbar.set_description("%d - Loss: %f"
#                                                   % (epoch, mu_loss))
#
#                        batch_pbar.update(1)
#                    #####
#                    if self.cv_data_gen is not None:
#                        self.model.eval()
#                        with tqdm(total=len(self.cv_data_gen), desc='CV::') as cv_pbar:
#                            with torch.no_grad():
#                                for cv_idx, cv_data_dict in enumerate(self.cv_data_gen):
#                                    cv_X = cv_data_dict['ecog_arr'].to(self.device)
#                                    cv_y = cv_data_dict['text_arr'].to(self.device)
#
#                                    cv_pred = self.model(cv_X)
#                                    cv_loss = self.criterion(cv_pred, cv_y)
#                                    self.cv_losses.append(cv_loss.detach().cpu().item())
#
#                                    cv_pbar.update(1)
#                                    cv_mean_loss = np.mean(self.cv_losses[-(1 + cv_idx):])
#                                    desc = "CV Loss: %.4f" % cv_mean_loss
#                                    cv_pbar.set_description(desc)
#
#                                if cv_mean_loss < best_cv:
#                                    self.best_model_state = copy_model_state(self.model)
#                                    self.best_model_epoch = epoch
#                                    desc = "CV Loss: %.4f [[NEW BEST]]" % cv_mean_loss
#                                    cv_pbar.set_description(desc)
#                                    best_cv = cv_mean_loss
#
#                        self.model.train()
#                epoch_pbar.update(1)
#
#        return self.losses

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


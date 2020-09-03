import attr
from tqdm.auto import tqdm
import numpy as np
import torch

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
                 channel_dim=1, unsqueeze_dim=1
                 #store_param_history=False
                 ):
        super(MultiChannelSincNN, self).__init__()
        self.channel_dim = channel_dim
        self.unsqueeze_dim = unsqueeze_dim
        #self.store_param_history = store_param_history
        sinc_kwargs = dict(in_channels=1, out_channels=num_bands,
                           kernel_size=kernel_size, sample_rate=fs,
                           min_low_hz=min_low_hz, min_band_hz=min_band_hz,
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


@attr.attrs
class Trainer:
    model = attr.ib()

    train_data_gen = attr.ib()
    optim_kwargs = attr.ib(dict(weight_decay=0.2, lr=0.001))
    cv_data_gen = attr.ib(None)
    epochs_trained = attr.ib(0)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def get_best_state(self):
        if self.best_model_state is not None:
            return self.best_model_state
        else:
            return self.copy_model_state(self.m)

    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None,
              batch_cb_delta=5):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        self.model = self.model.to(self.device)

        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.BCELoss()  # weight=torch.tensor(2))
        self.optim = torch.optim.Adam(self.model.parameters(), **self.optim_kwargs)
        self.losses = getattr(self, 'losses', list())
        self.cv_losses = getattr(self, 'cv_losses', list())
        best_cv = np.inf
        train_loss = 0
        with tqdm(total=n_epochs, desc='Train epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                self.model.train()
                with tqdm(total=len(self.train_data_gen), desc='-loss-') as batch_pbar:
                    for batch_idx, data_dict in enumerate(self.train_data_gen):
                        self.model.zero_grad()

                        # print("batch {i}")
                        ecog_arr = data_dict['ecog_arr']  # .to(self.device)
                        # ecog_arr = (ecog_arr/ecog_arr.abs().max(1, keepdim=True).values)

                        actuals = data_dict['text_arr']  # .to(self.device)
                        # print("running model")
                        m_output = self.model(ecog_arr)

                        self.optim.zero_grad()
                        loss = self.criterion(m_output, actuals)
                        # print("backward")
                        loss.backward()
                        self.optim.step()
                        l = loss.detach().cpu().item()

                        train_loss += l

                        self.losses.append(l)
                        mu_loss = np.mean(self.losses[-(batch_idx + 1):])
                        batch_pbar.set_description("%d - Loss: %f"
                                                   % (epoch, mu_loss))

                        batch_pbar.update(1)
                    #####
                    if self.cv_data_gen is not None:
                        self.model.eval()
                        with tqdm(total=len(self.cv_data_gen), desc='CV::') as cv_pbar:
                            with torch.no_grad():
                                for cv_idx, cv_data_dict in enumerate(self.cv_data_gen):
                                    cv_X = cv_data_dict['ecog_arr'].to(self.device)
                                    cv_y = cv_data_dict['text_arr'].to(self.device)

                                    cv_pred = self.model(cv_X)
                                    cv_loss = self.criterion(cv_pred, cv_y)
                                    self.cv_losses.append(cv_loss.detach().cpu().item())

                                    cv_pbar.update(1)
                                    cv_mean_loss = np.mean(self.cv_losses[-(1 + cv_idx):])
                                    desc = "CV Loss: %.4f" % cv_mean_loss
                                    cv_pbar.set_description(desc)

                                if cv_mean_loss < best_cv:
                                    self.best_model_state = copy_model_state(self.model)
                                    self.best_model_epoch = epoch
                                    desc = "CV Loss: %.4f [[NEW BEST]]" % cv_mean_loss
                                    cv_pbar.set_description(desc)
                                    best_cv = cv_mean_loss

                        self.model.train()
                epoch_pbar.update(1)



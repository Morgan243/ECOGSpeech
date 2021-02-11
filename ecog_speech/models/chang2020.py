import torch
import attr
from tqdm.auto import tqdm
from ecog_speech.models import base

class Encoder(torch.nn.Module):
    def __init__(self, n_input_channels, n_filters=100,
                 kernel_size=12, stride=12):
        super(Encoder, self).__init__()

        self.cnn_enc_m = torch.nn.Sequential(
            torch.nn.Conv1d(n_input_channels, n_filters,
                            kernel_size, stride=stride),
            torch.nn.BatchNorm1d(n_filters),
            torch.nn.ReLU(),
        )
        # input: (seq_len, batch, input_size)
        self.lstm_enc_a_m = torch.nn.LSTM(100, 400, 2, bidirectional=True,
                                          batch_first=True)
        # TODO: It's not clear if this is correct - 400x2 to maintain directions,
        #  or are they auto concat internally
        self.lstm_enc_b_m = torch.nn.LSTM(800, 400, 1, bidirectional=True,
                                          batch_first=True)

        self.mfcc_output_m = torch.nn.Sequential(
            torch.nn.Linear(800, 255),
            torch.nn.Linear(255, 13)
        )

    # TODO: option for not computing MFCC output during evaluation would be nice
    #   look into eval() and train() hooks for this?
    def forward(self, x):
        z = self.cnn_enc_m(x)
        # TODO: need to provide h_0, c_0 in some way, check paper
        #z = self
        #lstm_a_output, (lstm_a_h, lstm_a_c) = self.lstm_enc_a_m(z.permute(0, 2, 1))
        #lstm_b_output, (lstm_b_h, lstm_b_c) = self.lstm_enc_b_m(lstm_a_output)
        # Z is (batch, channels, seq_len)
        # input: (seq_len, batch, input_size)
        lstm_a_output, hidden_a_t = self.lstm_enc_a_m(z.permute(2, 0, 1))
        lstm_b_output, hidden_b_t = self.lstm_enc_b_m(lstm_a_output)

        mfcc_output = self.mfcc_output_m(lstm_a_output)

        return lstm_b_output, hidden_b_t, mfcc_output

class Decoder(torch.nn.Module):
    """
    From paper:
            The decoder RNN (gold rectangles) is initialized with the final state of the final
        layer of the encoder RNN. (In fact, this state is a concatenation of the final state
        of the forward encoder RNN with the first state of the backward encoder RNN,
        although both correspond to step M of the input sequence. Thus, the dimension of
        the decoder state is 800 = 400 × 2.) This RNN receives as input the preceding word,
        encoded one-hot and embedded in a 150-dimensional space with a fully connected
        layer of rectified-linear units (bluish boxes below). The decoder RNN is necessarily
        unidirectional, since it cannot be allowed to access future words. The output of the
        decoder RNN passes through a single matrix that projects the state into the space
        of words, with dimension equal to the vocabulary size.

    """
    def __init__(self, n_classes=150):
        super(Decoder, self).__init__()
        self.n_classes = n_classes

        # TODO: Is this right? Mapping from one-hot 150 to embdeded 150
        #   would think this would output a smaller dimension than the one hot
        self.word_embed_m = torch.nn.Sequential(
            torch.nn.Linear(n_classes, n_classes),
            torch.nn.ReLU()
            # Could put lstm here?
        )

        # The LSTM sees the previous class
        self.lstm_dec = torch.nn.LSTM(n_classes, 400, bidirectional=True, batch_first=True)

        self.output_m = torch.nn.Sequential(
            torch.nn.Linear(800, n_classes),
            torch.nn.PReLU()
        )

    def forward(self, x, hidden_t):
        z = self.word_embed_m(x)
        # X and Z shaped as (batch, features) - LSTM takes it differently
        _z = z.reshape(1, -1, self.n_classes)
        o = self.output_m(self.lstm_dec(_z, hidden_t)[0])
        return o.squeeze()

class SpeechDetectorDecoder(torch.nn.Module):
    def __init__(self, n_classes=1):
        super(SpeechDetectorDecoder, self).__init__()
        self.n_classes = n_classes
        # The LSTM sees the previous class
        self.lstm_dec = torch.nn.LSTM(n_classes, 400, bidirectional=True, batch_first=True)

        self.output_m = torch.nn.Sequential(
            torch.nn.Linear(800, n_classes),
            torch.nn.PReLU()
        )

    def forward(self, x, hidden_t):
        #return self.
        #z = self.word_embed_m(x)
        # X and Z shaped as (batch, features) - LSTM takes it differently
        #_z = z.reshape(1, -1, self.n_classes)
        o = self.output_m(self.lstm_dec(x, hidden_t)[0])
        return o.squeeze()


# 3 stack LSTM 150->100->50 hidden nodes in stack with 0.5 dropout between layers. Adam optimizer and only one extra bit of flair in the cross entropy loss function.
class ChangSpeechDetector(torch.nn.Module):
    def __init__(self, input_size, n_classes=3):
        super(ChangSpeechDetector, self).__init__()
        self.n_classes = n_classes
        lstm_kws = dict(batch_first=True)
        self.lstm_l = torch.nn.ModuleList([
            torch.nn.LSTM(input_size=input_size, hidden_size=150, **lstm_l),
            torch.nneLSTM(input_size=150, hidden_size=100, **lstm_l),
            torch.nneLSTM(input_size=100, hidden_size=50, **lstm_l)
            ])

    def forward(self, x):
        l_o = x
        for lstm_m in self.lstm_l:
            l_o, (l_h, l_c) = lstm_m(l_o)
        return l_o, (l_h, l_c)

@attr.attrs
class ChangTrainer(base.Trainer):
    def eval(self, epoch_i, dataloader):
        pass

    def train_inner_step(self, epoch_i, data_batch):
        encoder = self.model_map['encoder']
        ecog = data_batch['ecog_arr']
        lstm_b_output, hidden_b_t, mfcc_output = encoder(ecog)
        # Downsample (every 12th sample) and clip to only the first length
        # TODO: better way to handle the off-by-one or is something already incorrect?
        #mfcc = data_batch['mfcc'][:, ::12, :][:, :mfcc_output.shape[0], :]

        #mfcc_mse_l = torch.nn.functional.mse_loss(mfcc, mfcc_output.transpose(0, 1))


## WIP
@attr.attrs
class ChangMFCCTrainer(base.Trainer):
    def eval(self, epoch_i, dataloader):
        pass

    def train_inner_step(self, epoch_i, data_batch):
        encoder = self.model_map['encoder']
        ecog = data_batch['ecog_arr']
        lstm_b_output, hidden_b_t, mfcc_output = encoder(ecog)

        # Downsample (every 12th sample) and clip to only the first length
        # TODO: better way to handle the off-by-one or is something already incorrect?
        mfcc = data_batch['mfcc'][:, ::12, :][:, :mfcc_output.shape[0], :]
        mfcc_mse_l = torch.nn.functional.mse_loss(mfcc, mfcc_output.transpose(0, 1))


import torch
import attr
from tqdm.auto import tqdm

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
        self.lstm_enc_a_m = torch.nn.LSTM(100, 400, 2, bidirectional=True)
        # TODO: It's not clear if this is correct - 400x2 to maintain directions,
        #  or are they auto concat internally
        self.lstm_enc_b_m = torch.nn.LSTM(800, 400, 1, bidirectional=True)

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
        self.lstm_dec = torch.nn.LSTM(n_classes, 400, bidirectional=True)

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



## WIP
@attr.attrs
class Trainer:
    enc_model = attr.ib()
    dec_model = attr.ib()
    train_data_gen = attr.ib()

    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def train(self, n_epochs, epoch_callbacks=None, batch_callbacks=None,
              batch_cb_delta=5):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        self.enc_model = self.enc_model.to(self.device)
        self.dec_model = self.dec_model.to(self.device)

        self.enc_optim = torch.optim.Adam(self.enc_model.parameters())
        self.dec_optim = torch.optim.Adam(self.dec_model.parameters())

        with tqdm(total=n_epochs, desc='Train epoch') as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                with tqdm(total=len(self.train_data_gen), desc='-loss-') as batch_pbar:
                    for i, data_dict in enumerate(self.train_data_gen):
                        self.enc_model.zero_grad()
                        self.dec_model.zero_grad()

                        ecog_arr = data_dict['ecog_arr']
                        lstm_b_output, hidden_b_t, mfcc_output = self.enc_model(ecog_arr)

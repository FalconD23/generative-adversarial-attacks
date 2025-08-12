from torch import nn

from activation import Activation


class AttackLSTM(nn.Module):
    def __init__(self, hidden_dim=64, x_dim=1, activation_type='identity'):
        super().__init__()
        self.rnn_inp = nn.LSTM(x_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.4)
        self.act = Activation(activation_type)
        self.rnn_out = nn.LSTM(hidden_dim, 1, num_layers=3, batch_first=True, dropout=0.4)

    def forward(self, data):
        x, _ = self.rnn_inp(data)
        x = self.act(x)
        x, _ = self.rnn_out(x)
        return x

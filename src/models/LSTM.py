import torch
from torch import nn

from .utils import Activation
from .base_model import BaseModel


class LSTM(BaseModel):
    def __init__(
        self,
        hidden_dim: int = 50,
        n_layers: int = 1,
        x_dim: int = 1,
        output_dim: int = 1,
        dropout: float = 0.2,
        activation_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(
            x_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True
        )

        self.n_layers = n_layers
        self.fc1 = nn.Linear(hidden_dim * n_layers, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.final_activation = Activation(activation_type)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.rnn(data)

        # hidden have shape n_layers, n_onjects, hidden_dim
        n_objects = hidden.shape[1]

        hidden = hidden.transpose(1, 0)  # shape n_onjects, n_layers, hidden_dim

        hidden = hidden.reshape(
            n_objects,
            self.n_layers * self.hidden_dim,
        )  # shape n_onjects, n_layers * hidden_dim

        hidden = self.dropout(hidden)
        output = self.relu(self.fc1(hidden))
        output = self.fc2(self.dropout(output))
        return self.final_activation(output)
    


class GenLSTM(BaseModel):
    def __init__(
        self,
        hidden_dim: int = 50,
        n_layers: int = 1,
        x_dim: int = 1,
        dropout: float = 0.2,
        activation_type: str = "identity",
    ):
        super().__init__()

        self.rnn_inp = nn.LSTM(x_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.act = Activation(activation_type)
        self.rnn_out = nn.LSTM(hidden_dim, x_dim, num_layers=n_layers, batch_first=True, dropout=dropout)

    def forward(self, data):
        x, _ = self.rnn_inp(data)
        x = self.act(x)
        x, _ = self.rnn_out(x)
        return x

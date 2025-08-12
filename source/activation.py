from torch import nn


class Activation(nn.Module):
    def __init__(self, kind='identity'):
        super().__init__()
        if kind == 'identity':
            self.act = nn.Identity()
        elif kind == 'relu':
            self.act = nn.ReLU()
        elif kind == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f'Unknown activation {kind}')

    def forward(self, x):
        return self.act(x)
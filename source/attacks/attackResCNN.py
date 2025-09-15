from torch import nn
from tsai.models.all import ResCNN

from activation import Activation


class ResCNNModel(nn.Module):
    def __init__(self, x_dim, output_dim,
                 activation_type='identity',
                 rescnn_kwargs=None):
        super().__init__()
        self.x_dim = x_dim
        rescnn_kwargs = rescnn_kwargs or {}
        self.body = ResCNN(c_in=x_dim, c_out=output_dim, **rescnn_kwargs)
        self.fin = Activation(activation_type)

    def forward(self, x):
        if x.ndim == 3 and x.shape[1] != self.x_dim:
            x = x.transpose(1, 2)
        return self.fin(self.body(x))


class AttackCNN(nn.Module):
    def __init__(self, hidden_dim=128, x_dim=1, activation_type='tanh'):
        super().__init__()
        self.step_cnn = ResCNNModel(x_dim=x_dim, output_dim=hidden_dim, activation_type='identity')
        self.fc = nn.Linear(hidden_dim, x_dim)
        self.act = Activation(activation_type)

    def forward(self, x):
        B, L, C = x.shape
        x_flat = x.contiguous().view(B * L, 1, C)
        h = self.step_cnn(x_flat)
        h = h.view(B, L, -1)
        return self.fc(self.act(h))
        # return self.act(h)

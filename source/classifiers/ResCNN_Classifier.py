from torch import nn
from tsai.models.ResCNN import ResCNN

from activation import Activation


class ResCNNClassifier(nn.Module):
    def __init__(self, n_classes, x_dim=1, activation_type='identity', rescnn_kwargs=None):
        super().__init__()
        self.x_dim = x_dim
        rescnn_kwargs = rescnn_kwargs or {}

        # NN-backbone: tsai.models.ResCNN
        self.body = ResCNN(c_in=x_dim, c_out=n_classes, **rescnn_kwargs)
        self.fin = Activation(activation_type)

    def forward(self, x):
        # (B,L,C) or (B,C,L) —> (B,C,L)
        if x.ndim == 3 and x.shape[1] != self.x_dim:   # (B,L,C)
            x = x.transpose(1, 2)
        return self.fin(self.body(x))                  # logits
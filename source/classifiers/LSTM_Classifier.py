from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_classes, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

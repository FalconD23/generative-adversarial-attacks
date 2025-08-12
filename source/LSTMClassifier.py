import torch
from torch import nn
from early_stopping import EarlyStopping


class LSTMClassifier(nn.Module):
    def __init__(self, n_classes, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train_lstm_classifier(n_classes, train_dl, train_size, test_dl, test_size, epochs, lr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMClassifier(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper_cls = EarlyStopping(patience=4, mode="min")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0., 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
        train_acc = correct / train_size

        model.eval()
        val_loss, val_correct = 0., 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
        val_loss /= test_size
        val_acc = val_correct / test_size
        print(f'Epoch {epoch:2d} | train_acc {train_acc:.3f} | val_loss {val_loss:.4f} | val_acc {val_acc:.3f}')

        if stopper_cls.step(val_loss):
            print(f'⏹ Early stopping classifier at epoch {epoch}')
            break
    return model

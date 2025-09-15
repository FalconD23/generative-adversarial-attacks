import torch
from torch import nn

from LSTM_Classifier import LSTMClassifier


class LongLSTMClassifier(nn.Module):
    """
    Делит последовательность на N равных частей и прогоняет каждую
    через собственный экземпляр LSTMClassifier. Финальный ответ —
    наиболее встречающаяся метка (mode) среди N предсказаний.

    Parameters
    ----------
    n_classes : int
        Число классов.
    n_splits : int
        Сколько кусков делать из входной последовательности.
    input_size, hidden_size, num_layers, dropout
        Параметры, которые передаются во внутренние LSTMClassifier’ы.
    """
    def __init__(self,
                 n_classes: int,
                 n_splits: int = 5,
                 input_size: int = 1,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        assert n_splits >= 1, "n_splits должно быть ≥ 1"
        self.n_splits = n_splits

        # создаём N независимых LSTMClassifier’ов
        self.sub_nets = nn.ModuleList([
            LSTMClassifier(n_classes,
                           input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
            for _ in range(n_splits)
        ])

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     x : (B, L, C) – батч длинных последовательностей
    #     Возвращает (B,) с итоговой меткой, либо (B, n_classes) с логитами,
    #     если нужно (зависит от задачи).
    #     """
    #     B, L, C = x.shape
    #     # длина одного куска (последний может быть короче)
    #     chunk_len = (L + self.n_splits - 1) // self.n_splits

    #     logits_list = []
    #     for i in range(self.n_splits):
    #         start = i * chunk_len
    #         end   = min(start + chunk_len, L)
    #         if start >= L:                       # если кусок «вышел» за предел – дублируем последний
    #             part = x[:, -chunk_len:, :]
    #         else:
    #             part = x[:, start:end, :]        # (B, chunk, C)
    #         # прогоняем через свой LSTM
    #         logits = self.sub_nets[i](part)      # (B, n_classes)
    #         logits_list.append(logits)

    #     # -> (n_splits, B, n_classes)
    #     stacked = torch.stack(logits_list, dim=0)
    #     # предсказанные метки каждой подсети, shape (n_splits, B)
    #     preds = stacked.argmax(-1)

    #     # мода по оси n_splits
    #     mode_preds = []
    #     for b in range(B):
    #         # Counter возвращает [(метка, частота), ...]; берём самую частую
    #         most_common = Counter(preds[:, b].tolist()).most_common(1)[0][0]
    #         mode_preds.append(most_common)
    #     mode_preds = torch.tensor(mode_preds, device=x.device)  # (B,)

    #     return mode_preds        # можно вернуть ещё stacked.mean(0) как «сглаженные» логиты

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        chunk_len = (L + self.n_splits - 1) // self.n_splits

        logits_per_split = []
        for i in range(self.n_splits):
            start = i * chunk_len
            end   = min(start + chunk_len, L)
            part  = x[:, start:end, :] if start < L else x[:, -chunk_len:, :]
            logits_per_split.append(self.sub_nets[i](part))   # (B, n_classes)

        # (n_splits, B, n_classes) → усредняем / суммируем
        logits = torch.stack(logits_per_split, dim=0).mean(0)   # (B, n_classes)
        return logits
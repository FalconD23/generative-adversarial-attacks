import torch
from torch import nn

from attackLSTM import AttackLSTM


class AttackLongLSTM(nn.Module):
    """
    Разбивает вход (B, L, C) на n_splits кусков по временной оси,
    каждый кусок ― своя attack_LSTM. На выходе тензор той же формы.
    """

    def __init__(
            self,
            n_splits: int = 4,
            hidden_dim: int = 64,
            x_dim: int = 1,
            activation_type: str = "identity",
            dropout: float = 0.25,
    ):
        super().__init__()
        assert n_splits >= 1, "n_splits должно быть ≥ 1"
        self.n_splits = n_splits

        # N независимых атакующих LSTM
        self.sub_atks = nn.ModuleList([
            AttackLSTM(hidden_dim=hidden_dim,
                       x_dim=x_dim,
                       activation_type=activation_type,
                       dropout=dropout)
            for _ in range(n_splits)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, C)
        out: (B, L, C) – клеенный результат всех под-атак.
        """
        B, L, _ = x.shape
        chunk_len = (L + self.n_splits - 1) // self.n_splits  # округление вверх

        chunks_out = []
        for i in range(self.n_splits):
            start = i * chunk_len
            end = min(start + chunk_len, L)

            # если вышли за пределы – дублируем последний срез
            part = x[:, start:end, :] if start < L else x[:, -chunk_len:, :]

            # обрабатываем своим attack_LSTM
            chunks_out.append(self.sub_atks[i](part))

        # усечение до исходной длины L (последний кусок мог быть длиннее)
        out = torch.cat(chunks_out, dim=1)[:, :L, :]
        return out

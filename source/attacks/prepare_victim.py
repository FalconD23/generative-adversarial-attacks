from torch import nn


def prepare_victim_for_input_grad(victim: nn.Module):
    # 1) общий eval + заморозка весов
    victim.eval()
    for p in victim.parameters():
        p.requires_grad_(False)

    # 2) RNN в train(True), без dropout
    for m in victim.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            m.train(True)
            if hasattr(m, 'dropout'):
                m.dropout = 0.0
            # иногда полезно обновить внутренние буферы для cuDNN
            try:
                m.flatten_parameters()
            except Exception:
                pass

    # 3) отключаем стохастику в явных Dropout, BN оставляем в eval
    for m in victim.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            m.eval()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

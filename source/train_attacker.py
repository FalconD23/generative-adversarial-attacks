import torch
import torch.nn.functional as F

from source.early_stopping import EarlyStopping


def train_attacker(attacker, victim, loader, eps, epochs=50, lr=1e-4, alpha_l2=1e-3, device='cpu', patience=4,
                   clamp=False, new_loss=False):
    attacker.to(device)
    opt = torch.optim.Adam(attacker.parameters(), lr)
    stopper = EarlyStopping(patience=patience, mode="max")
    for ep in range(1, epochs + 1):
        attacker.train()
        run_vloss, run_acc, n = 0., 0., 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            delta = eps * torch.tanh(attacker(x))
            x_adv = x + delta
            if clamp:
                x_adv = torch.clamp(x_adv, -1, 1)
            logits = victim(x_adv)
            vloss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean().item()
            reg = alpha_l2 * (delta ** 2).mean()
            if new_loss:
                loss = F.cross_entropy(logits, 1 - y) + reg
            else:
                loss = -vloss + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_vloss += vloss.item() * x.size(0)
            run_acc += acc * x.size(0)
            n += x.size(0)
        val_loss = run_vloss / n
        print(f'Epoch {ep:02d} | victim‑loss {val_loss:.4f} | acc {run_acc / n:.4f}')
        if stopper.step(val_loss):
            print(f'⏹ Early stopping at epoch {ep}')
            break
    torch.save(attacker.state_dict(), 'attacker_maxloss.pth')

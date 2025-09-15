import torch
import torch.nn.functional as F

from source.early_stopping import EarlyStopping


def train_attacker(attacker,
                   victim,
                   loader,
                   eps,
                   epochs=50,
                   lr=1e-4,
                   alpha_l2=1e-3,
                   device='cpu',
                   patience=4,
                   clamp=False,
                   debug=False,
                   new_loss=False
                   ):
    attacker.to(device)
    opt = torch.optim.Adam(attacker.parameters(), lr)
    stopper = EarlyStopping(patience=patience, mode="max")
    for ep in range(1, epochs + 1):
        attacker.train()
        run_vloss, run_acc, n = 0., 0., 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # * tanh regularization
            # delta = eps * torch.tanh(atk_model(x))

            # * l2-proj regularization
            raw_delta = attacker(x)
            norms = raw_delta.norm(p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1)
            delta = eps * raw_delta / (norms + 1e-12)

            # * sign regularization
            # delta = eps * torch.sign(atk_model(x))

            x_adv = x + delta
            logits = victim(x_adv)
            vloss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean().item()
            reg = alpha_l2 * (delta ** 2).mean()
            if new_loss:
                loss = F.cross_entropy(logits, 1 - y) + reg
            else:
                loss = -vloss + reg
            if debug:
                print(f'\nx={((x[0].detach().cpu() ** 2).mean()) ** 0.5}')
                print(f'delta={(delta[0].detach().cpu().norm(p=2).item())}')
                print(f'y={y.detach().cpu()[0]}|logits={logits.detach().cpu().argmax(1)[0]}')
                print(f'vloss = {vloss}| reg = {reg}')

                with torch.no_grad():
                    # ----- (2) % L2-нормы входа -----
                    # L2 посчитаем по каждому объекту, затем усредним
                    l2_delta = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1)
                    l2_x_adv = torch.norm(x_adv.view(x_adv.size(0), -1), p=2, dim=1) + 1e-8
                    l2_x = torch.norm(x.view(x.size(0), -1), p=2, dim=1) + 1e-8
                    perc_l2_x_adv = (l2_delta / l2_x_adv * 100).mean().item()
                    perc_l2_x = (l2_x_adv / l2_x * 100).mean().item()

                print(f'\nL2:Δ/x_adv: {perc_l2_x_adv:6.2f}%   x_adv/x: {perc_l2_x:6.2f}%')

            opt.zero_grad()
            loss.backward()
            opt.step()
            run_vloss += vloss.item() * x.size(0)
            run_acc += acc * x.size(0)
            n += x.size(0)

        val_loss = run_vloss / n
        print(f'\nEpoch {ep:02d} | victim‑loss {val_loss:.4f} | acc {run_acc / n:.4f}')
        if stopper.step(val_loss):
            print(f'⏹ Early stopping at epoch {ep}')
            break
    torch.save(attacker.state_dict(), 'attacker_maxloss.pth')
    return val_loss, run_acc / n

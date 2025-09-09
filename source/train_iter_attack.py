import torch
from torch import nn
import torch.nn.functional as F

from early_stopping import EarlyStopping
from iter_model_attack import IterModelAttack


@torch.no_grad()
def _freeze_(m: nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)


def train_attack_iter(
        attacker: nn.Module,
        victim: nn.Module,
        loader,
        eps=0.5,
        steps=10,
        alpha=None,
        epochs=50,
        lr=1e-4,
        alpha_l2=1e-3,
        lambda_disc=0.0,
        disc: nn.Module | None = None,
        device="cpu",
        patience=4,
        data_clamp=None,  # как раньше
        rand_init=True,
        use_sign=False,
        equal_eps=False,
        bpda=True,
        verbose=True,
        # ---- новые флаги (все опциональны, дефолты сохраняют прежнее поведение) ----
        proj: str = "none",  # "none" | "linf" | "l2"
        proj_equal_eps: bool = False,
        momentum_mu: float = 0.0,  # MI-FGSM
        step_normalize: str | None = None,  # "meanabs" | "l2" | "linf" | None
        step_noise_std: float = 0.0,
        victim_eval: bool = True,
        grad_clip: float | None = None,
):
    attacker.to(device).train()
    victim.to(device)
    if victim_eval:
        victim.eval()
    _freeze_(victim)
    if disc is not None:
        disc.to(device).eval()
        _freeze_(disc)

    iter_attack = IterModelAttack(
        attacker=attacker, eps=eps, n_iter=steps, alpha=alpha,
        clamp=data_clamp, rand_init=rand_init, use_sign=use_sign,
        equal_eps=equal_eps, bpda=bpda,
        proj=proj, proj_equal_eps=proj_equal_eps,
        data_clamp=data_clamp,
        momentum_mu=momentum_mu, step_normalize=step_normalize,
        step_noise_std=step_noise_std,
    ).to(device)

    opt = torch.optim.Adam(attacker.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience, mode="max")  # максимизируем victim-loss

    best_val, best_acc = float("-inf"), 0.0
    for ep in range(1, epochs + 1):
        attacker.train()
        run_vloss, run_acc, n = 0.0, 0.0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = iter_attack(x)

            # ---- средняя норма по батчу ----
            if verbose:
                with torch.no_grad():
                    delta = (x_adv - x).detach()
                    # L2 по объектам, затем среднее по батчу
                    l2_mean = delta.view(delta.size(0), -1).norm(p=2, dim=1).mean().item()
                    # L_inf по объектам, затем среднее по батчу
                    linf_mean = delta.view(delta.size(0), -1).abs().amax(dim=1).mean().item()
                    x_mean = x.view(x.size(0), -1).norm(p=2, dim=1).mean().item()
                    x_adv_mean = x_adv.view(x_adv.size(0), -1).norm(p=2, dim=1).mean().item()
                print(f"[Δ] mean ||δ||_2 = {l2_mean:.3f} | mean ||δ||_∞ = {linf_mean:.3f}")
                print(f"[Δ] mean ||x||_2 = {x_mean:.3f} | mean ||x_adv||_2 = {x_adv_mean:.3f}")

            logits = victim(x_adv)
            vloss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean().item()

            loss_disc = torch.tensor(0.0, device=device)
            if disc is not None:
                d_out = disc(x_adv)
                if d_out.dim() == 1 or d_out.size(-1) == 1:
                    target = torch.zeros_like(d_out)
                    loss_disc = nn.BCEWithLogitsLoss()(d_out, target)
                else:
                    target = torch.zeros(d_out.size(0), dtype=torch.long, device=device)
                    loss_disc = nn.CrossEntropyLoss()(d_out, target)

            delta = (x_adv - x)
            reg = alpha_l2 * (delta ** 2).mean()

            loss = -(vloss) + lambda_disc * loss_disc + reg

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(attacker.parameters(), grad_clip)
            opt.step()

            run_vloss += vloss.item() * x.size(0)
            run_acc += acc * x.size(0)
            n += x.size(0)

        val_loss = run_vloss / n
        val_acc = run_acc / n
        if verbose:
            print(f"Epoch {ep:02d} | victim-loss {val_loss:.4f} | acc {val_acc:.4f}")

        if stopper.step(val_loss):
            if verbose: print(f"⏹ Early stopping at epoch {ep}")
            break

    torch.save(attacker.state_dict(), 'attacker_iter_unrolled.pth')
    return val_loss, val_acc

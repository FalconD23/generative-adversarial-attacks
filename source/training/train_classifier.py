from collections import defaultdict
import torch
import torch.nn.functional as F

from early_stopping import EarlyStopping


def train_classifier(model,
                     train_loader,
                     val_loader=None,
                     epochs=50,
                     lr=1e-3,
                     weight_decay=0.,
                     device="cpu",
                     patience=5,
                     verbose_every=3):
    """
    returns: dict(history), best_val_loss, best_state_dict
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=patience, mode="min") if val_loader else None
    history = defaultdict(list)
    best_wts = model.state_dict()
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # ----- train -----
        model.train()
        loss_sum, acc_sum, n = 0., 0., 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += x.size(0)

        train_loss = loss_sum / n
        train_acc = acc_sum / n
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ----- validation -----
        if val_loader:
            model.eval()
            with torch.no_grad():
                loss_sum, acc_sum, n = 0., 0., 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss_sum += loss.item() * x.size(0)
                    acc_sum += (logits.argmax(1) == y).float().sum().item()
                    n += x.size(0)

            val_loss = loss_sum / n
            val_acc = acc_sum / n
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                best_wts = {k: v.cpu() for k, v in model.state_dict().items()}

            if stopper and stopper.step(val_loss):
                if verbose_every:
                    print(f"⏹ Early stopping on epoch {ep:02d}")
                break

        if verbose_every and ep % verbose_every == 0:
            msg = f"Epoch {ep:02d}: train_loss={train_loss:.4f} acc={train_acc:.4f}"
            if val_loader:
                msg += f" | val_loss={val_loss:.4f} acc={val_acc:.4f}"
            print(msg)

    model.load_state_dict(best_wts)
    return history, best_val, best_wts

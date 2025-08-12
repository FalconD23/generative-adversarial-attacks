import torch


def fooling_rate(model, loader, attack, device='cpu'):
    # model.eval()
    fooled, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds_orig = model(x).argmax(dim=1)
    fooled, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds_orig = model(x).detach().argmax(dim=1)
        x_adv = attack(model, x, y)
        preds_adv = model(x_adv).detach().argmax(dim=1)
        fooled += (preds_adv != preds_orig).sum().item()
        total += x.size(0)
    return fooled / total


def efficiency_rate(model, loader, attack, device='cpu'):
    # model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y)

        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)

        true_positives += ((preds_adv == 1) & (y == 1)).sum().item()
        false_positives += ((preds_adv == 1) & (y == 0)).sum().item()
        false_negatives += ((preds_adv == 0) & (y == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return 1 - f1


def accuracy_after_attack(model, loader, attack, device='cpu'):
    # model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y)

        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)

        correct += (preds_adv == y).sum().item()
        total += y.size(0)

    accuracy = correct / total
    return accuracy

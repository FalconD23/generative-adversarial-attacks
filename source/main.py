import random
import os
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from power_cons import load_powercons, PowerConsDataset
from LSTMClassifier import train_lstm_classifier
from attacks import *
from attackLSTM import AttackLSTM
from attackResCNN import AttackCNN
from train_attacker import train_attacker
from metrics import *

SEED = 2
EPS = 0.5
BATCH_SIZE = 64
TRAIN_PATH = Path('PowerCons_TRAIN.tsv')
TEST_PATH = Path('PowerCons_TEST.tsv')


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, y_train, classes_ = load_powercons(TRAIN_PATH)
    X_test, y_test, _ = load_powercons(TEST_PATH)
    print('Train shape:', X_train.shape, 'Test shape:', X_test.shape, 'n_classes:', len(classes_))

    g = torch.Generator()
    g.manual_seed(SEED)
    train_ds = PowerConsDataset(X_train, y_train)
    test_ds = PowerConsDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, worker_init_fn=seed_worker,
                          generator=g)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print("LSTM classifier training")
    model = train_lstm_classifier(2, train_dl, len(train_ds), test_dl, len(test_ds), epochs=20, lr=1e-3)

    model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)

    print("LSTM attacker training")
    attacker_LSTM = AttackLSTM(hidden_dim=64, x_dim=1, activation_type='tanh').to(device)
    train_attacker(attacker_LSTM, model, train_dl, eps=EPS, epochs=20, lr=1e-4, alpha_l2=0,#1e-3,
                   device=device, new_loss=True)

    print("CNN attacker training")
    attacker_cnn = AttackCNN().to(device)
    train_attacker(attacker_cnn, model, train_dl,
                   eps=EPS, epochs=20, lr=1e-4, alpha_l2=0,#1e-3,
                   device=device, new_loss=True)

    fgsm_attack = FGSMAttack(EPS)
    # ifgsm_attack = iFGSMAttack(eps=EPS, n_iter=20)
    model_attack_lstm = ModelBasedAttack(attacker_LSTM, EPS)
    model_attack_cnn = ModelBasedAttack(attacker_cnn, EPS)

    print(f'eps = {EPS}')

    attacks = {'Unattacked': lambda m, x, y: x, 'FGSM': fgsm_attack,
               'LSTM': model_attack_lstm, 'CNN': model_attack_cnn}

    for name, atk in attacks.items():
        print(name)
        fl_rate = fooling_rate(model, test_dl, atk, device=device)
        ef_rate = efficiency_rate(model, test_dl, atk, device=device)
        acc = accuracy_after_attack(model, test_dl, atk, device=device)
        print(
            f'fooling rate:  {fl_rate:.3f},  efficiency metric:  {ef_rate:.3f},  target accuracy after attack:  {acc:.3f}')


if __name__ == "__main__":
    main()

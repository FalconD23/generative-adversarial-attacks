import random
import os
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from power_cons import load_powercons, PowerConsDataset
from LSTM_Classifier import LSTMClassifier
from ResCNN_Classifier import ResCNNClassifier
from train_classifier import train_classifier
from attacks import *
from attackLSTM import AttackLSTM
from attackResCNN import AttackCNN
from attackPatchTST import AttackPatchTST
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
    n_classes = len(classes_)
    print('Train shape:', X_train.shape, 'Test shape:', X_test.shape, 'n_classes:', n_classes)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_ds = PowerConsDataset(X_train, y_train)
    test_ds = PowerConsDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True, worker_init_fn=seed_worker,
                          generator=g)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Train LSTM classifier
    print("LSTM classifier training")
    clf_LSTM = LSTMClassifier(n_classes, hidden_size=50, num_layers=1)

    history, _, _ = train_classifier(
        clf_LSTM,
        train_dl,
        val_loader=test_dl,
        epochs=50,
        lr=1e-3,
        weight_decay=0.,
        device=device,
        patience=4,
        verbose_every=1
    )

    if device == 'cpu':
        clf_LSTM.to(device).eval()
    elif device == 'cuda':
        clf_LSTM.to(device).train()

    for p in clf_LSTM.parameters():
        p.requires_grad_(False)

    # Train ResCNN classifier
    clf_resCNN = ResCNNClassifier(n_classes=n_classes, x_dim=1).to(device)

    history, best_val, best_state = train_classifier(
        clf_resCNN,
        train_loader=train_dl,
        val_loader=test_dl,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        patience=10,
        verbose_every=1
    )

    clf_resCNN.to(device).eval()
    for p in clf_resCNN.parameters():
        p.requires_grad_(False)

    # Train LSTM attack
    print("LSTM attacker training")
    atk_LSTM = AttackLSTM(hidden_dim=128, dropout=0.6, x_dim=1, activation_type='tanh').to(device)

    eps_LSTM = 1.371353
    train_attacker(atk_LSTM,
                   clf_LSTM,
                   train_dl,
                   eps=eps_LSTM,
                   epochs=20,
                   lr=0.01749500,
                   alpha_l2=0.00068663089588,
                   device=device,
                   patience=9
                   )

    # Train ResCNN attack
    print("CNN attacker training")
    atk_resCNN = AttackCNN(hidden_dim=64, x_dim=1, activation_type='tanh').to(device)

    ''' 
    #! training is very sensitive to random-seed
    weights were obtained with the parameters:
    {'eps': 1.9910549963365909, 'lr': 0.0021041627898080928,
    'alpha_l2': 4.549583575912758e-05, 'patience': 6}
    '''
    # weights_path = 'weights/surr_resCNNfc_CPU_0.28.pth'
    # atk_resCNN.load_state_dict(torch.load(weights_path, map_location=device))

    eps_resCNN = 1.9910549963365909
    train_attacker(atk_resCNN, clf_resCNN, train_dl,
                   eps=eps_resCNN,
                   epochs=50, lr=0.00021041627898080928, alpha_l2=4.549583575912758e-05,
                   device=device, patience=10, debug=False)

    # Train PatchTST attack
    patch_params = dict(
        seq_len=144,
        n_layers=3,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        dropout=0.4,
        attn_dropout=0.0,
        patch_len=28,
        stride=24,
        padding_patch=True,
        revin=True,
        affine=False,
        individual=False,
        subtract_last=False,
        decomposition=False,
        kernel_size=25,
        activation="gelu",
        norm="BatchNorm",
        pre_norm=False,
        res_attention=True,
        store_attn=False,
    )

    atk_PatchTST = AttackPatchTST(
        hidden_dim=256,
        x_dim=1,
        activation_type="tanh",
        patch_kwargs=patch_params,
    )

    eps_PatchTST = 1.52738926
    train_attacker(atk_PatchTST, clf_resCNN, train_dl,
                   eps=eps_PatchTST,
                   epochs=50, lr=0.000202314, alpha_l2=0.000114732,
                   device=device, patience=8, debug=False)

    # Comparsion
    fgsm_attack = FGSMAttack(EPS)

    mba_LSTM = ModelBasedAttack(atk_LSTM, EPS)
    mba_resCNN = ModelBasedAttack(atk_resCNN, EPS)
    mba_PatchTST = ModelBasedAttack(atk_PatchTST, EPS)
    estimation_dl = train_dl

    attacks = {'Unattacked': lambda model, x, y: x,
               'FGSM': fgsm_attack,
               'iFGSM10nonrand02': iFGSMAttack(eps=EPS, n_iter=10, rand_init=False, momentum=0.2),
               'LSTM': mba_LSTM,
               'resCNN': mba_resCNN,
               'PatchTST': mba_PatchTST}

    def test_attacks(classification_model):
        for name, atk in attacks.items():
            fl_rate = fooling_rate(classification_model, estimation_dl, atk, device=device)
            ef_rate = efficiency_rate(classification_model, estimation_dl, atk, device=device)
            acc = accuracy_after_attack(classification_model, estimation_dl, atk, device=device)
            print(
                f'{name:<12} fooling rate:  {fl_rate:.3f},  efficiency metric:  {ef_rate:.3f},  target accuracy after attack:  {acc:.3f}')

    test_attacks(clf_LSTM)
    test_attacks(clf_resCNN)

    # TODO: transfer visualization and optuna


if __name__ == "__main__":
    main()

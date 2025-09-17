import random
import os
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from power_cons import load_powercons, PowerConsDataset
from LSTM_Classifier import LSTMClassifier
from ResCNN_Classifier import ResCNNClassifier
from iter_model_attack import IterModelAttack
from long_LSTM_classifier import LongLSTMClassifier
from prepare_victim import prepare_victim_for_input_grad
from attack_long_LSTM import AttackLongLSTM
from train_iter_attack import train_attack_iter
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
long = False
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
    if long:
        clf_LSTM = LongLSTMClassifier(
            n_classes=n_classes,
            n_splits=10,
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        ).to(device)
    else:
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
    if long:
        atk_LSTM = AttackLongLSTM(
            n_splits=11,
            hidden_dim=64,
            x_dim=1,
            dropout=0.25,
        ).to(device)
    else:
        atk_LSTM = AttackLSTM(hidden_dim=128, dropout=0.6, x_dim=1, activation_type='tanh').to(device)

    # eps_LSTM = 1.371353
    train_attacker(atk_LSTM,
                   clf_LSTM,
                   train_dl,
                   eps=EPS,
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

    # eps_resCNN = 1.9910549963365909
    train_attacker(atk_resCNN, clf_resCNN, train_dl,
                   eps=EPS,
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
    ).to(device)

    # eps_PatchTST = 1.52738926
    train_attacker(atk_PatchTST, clf_resCNN, train_dl,
                   eps=EPS,
                   epochs=50, lr=0.000202314, alpha_l2=0.000114732,
                   device=device, patience=8, debug=False)

    # Iter attack

    attacker = AttackPatchTST(hidden_dim=256, x_dim=1, activation_type='tanh',
                              patch_kwargs=patch_params).to(device)
    # attacker.load_state_dict(atk_PatchTST.state_dict())

    target = clf_resCNN
    prepare_victim_for_input_grad(target)

    disc = None

    best = {'eps': 0.3951928744395862,
            'lr': 0.0006180267696054032,
            'alpha_l2': 0.00019513735579565044,
            'steps': 13, 'use_alpha_explicit': False,
            'proj': 'none', 'proj_equal_eps': True, 'rand_init': True, 'bpda': False,
            'use_sign': False, 'momentum_mu': 0.8785765938958393, 'step_normalize': None,
            'step_noise_std': 0.004224670729218012, 'victim_eval': True, 'hidden_dim': 128}
    best['alpha'] = None

    alpha_explicit = None

    val_loss, val_acc = train_attack_iter(
        attacker=attacker,
        victim=target,
        loader=train_dl,

        eps=EPS,
        # eps=best["eps"],
        steps=best["steps"],
        alpha=alpha_explicit,
        epochs=10,
        lr=best["lr"],
        alpha_l2=best["alpha_l2"],

        lambda_disc=0.0,  # disc=None → без компоненты дискриминатора
        disc=disc,
        device=device,
        patience=8,

        # Базовые флаги совместимости
        data_clamp=None,
        rand_init=best["rand_init"],
        use_sign=best["use_sign"],
        equal_eps=best["proj_equal_eps"],  # для совместимости; реальный флаг ниже
        bpda=best["bpda"],
        verbose=True,

        # Новые расширенные флаги из обновлённого тренера
        proj=best["proj"],  # "none"
        proj_equal_eps=best["proj_equal_eps"],  # False
        momentum_mu=best["momentum_mu"],  # MI-FGSM momentum
        step_normalize=best["step_normalize"],  # "linf"
        step_noise_std=best["step_noise_std"],  # небольшой шум шага
        victim_eval=best["victim_eval"],  # False (нужно для cudnn-RNN backward)
        grad_clip=None,  # безопасное ограничение градиента
    )

    print(f"[done] val_loss={val_loss:.4f}  |  val_acc(after attack)={val_acc:.4f}")

    iter_attack = IterModelAttack(
        attacker=attacker, eps=best['eps'], n_iter=best['steps'], alpha=None,
        clamp=None, rand_init=best['rand_init'], use_sign=best['use_sign'],
        equal_eps=best['proj_equal_eps'], bpda=best['bpda'],
        proj=best['proj'], proj_equal_eps=best['proj_equal_eps'],
        data_clamp=None,
        momentum_mu=best['momentum_mu'], step_normalize=None,
        step_noise_std=best['step_noise_std'],
    ).to(device)

    eps_iter = best['eps']

    # Comparsion
    fgsm_attack = FGSMAttack(EPS)
    ifgsm_attack = iFGSMAttack(eps=EPS, n_iter=10, rand_init=False, momentum=0.2)
    pgd_attack = PGDAttack(EPS, n_iter=10)

    mba_LSTM = ModelBasedAttack(atk_LSTM, EPS)
    mba_resCNN = ModelBasedAttack(atk_resCNN, EPS)
    mba_PatchTST = ModelBasedAttack(atk_PatchTST, EPS)

    mba_iter = ModelBasedAttack(iter_attack, EPS, is_iter=True)
    estimation_dl = train_dl

    attacks = {'Unattacked': lambda model, x, y: x,
               'FGSM': fgsm_attack,
               'iFGSM': ifgsm_attack,
               'PGD': pgd_attack,
               'LSTM': mba_LSTM,
               'resCNN': mba_resCNN,
               'PatchTST': mba_PatchTST,
               'iter': mba_iter}

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

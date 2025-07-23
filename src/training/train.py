import os
from functools import partial
from typing import Any, Dict, List, Tuple
import copy

import numpy as np
import optuna
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from optuna.trial import Trial
from torch.utils.data import DataLoader
from clearml import Task

from src.attacks import BaseIterativeAttack
from src.attacks.attack_scheduler import AttackScheduler
from src.config import (
    get_attack,
    get_attack_scheduler,
    get_criterion,
    get_model,
    get_optimizer,
    get_scheduler,
)
from src.estimation import ClassifierEstimator
from src.utils import (
    collect_default_params,
    fix_seed,
    get_optimization_dict,
    update_dict_params,
    update_params_with_attack_params,
)


class EarlyStopper:
    def __init__(
        self,
        patience: int = 1,
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        multiclass: bool = False,
        train_self_supervised: bool = True,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.estimator = ClassifierEstimator()
        self.n_epochs = n_epochs
        self.early_stop_patience = early_stop_patience

        self.device = device
        self.multiclass = multiclass
        self.print_every = print_every
        self.train_self_supervised = train_self_supervised

        self.logger = logger
        self.dict_logging = dict()

        self.disc_trainer = False

    @staticmethod
    def initialize_with_params(
        model_name: str = "LSTM",
        model_params: Dict = None,
        criterion_name: str = "BCELoss",
        criterion_params: Dict = None,
        optimizer_name: str = "Adam",
        optimizer_params: Dict = None,
        scheduler_name: str = "None",
        scheduler_params: Dict = None,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        seed: int = 0,
        multiclass: bool = False,
        train_self_supervised: bool = True,
    ):
        fix_seed(seed)
        if model_params == "None" or not model_params:
            model_params = {}
        if criterion_params == "None" or not criterion_params:
            criterion_params = {}
        if optimizer_params == "None" or not optimizer_params:
            optimizer_params = {}
        if scheduler_params == "None" or not scheduler_params:
            scheduler_params = {}

        model = get_model(model_name, model_params, device=device)
        criterion = get_criterion(criterion_name, criterion_params)
        optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)
        scheduler = get_scheduler(scheduler_name, optimizer, scheduler_params)
        return Trainer(
            model,
            criterion,
            optimizer,
            scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
            multiclass=multiclass,
            train_self_supervised=train_self_supervised,
        )

    @staticmethod
    def initialize_with_optimization(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optuna_params: Dict,
        const_params: Dict,
    ):
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                Trainer.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                optim_metric=optuna_params["optim_metric"],
                const_params=const_params,
                train_loader=train_loader,
                valid_loader=valid_loader,
            ),
            n_trials=optuna_params["n_trials"],
        )

        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        print("DEFAULT", default_params)
        best_params = study.best_params.copy()
        print("BEST", best_params)
        best_params = update_dict_params(default_params, best_params)

        best_params = update_params_with_attack_params(const_params, best_params)

        print("Best parameters are - %s", best_params)
        return Trainer.initialize_with_params(**best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        optim_metric: str,
        const_params: Dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> float:
        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)

        initial_model_parameters = update_params_with_attack_params(
            const_params, initial_model_parameters
        )

        model = Trainer.initialize_with_params(**initial_model_parameters)
        last_epoch_metrics = model.train_model(train_loader, valid_loader)

        return last_epoch_metrics[optim_metric]

    def _init_logging(self, metric_names: List[str]) -> None:
        self.metric_names = metric_names
        self.dict_logging = {
            "train": {metric: [] for metric in self.metric_names},
            "test": {metric: [] for metric in self.metric_names},
        }
        self.print_line = "Epoch {} train loss: {}; acc_train {}; test loss: {}; acc_test {}; f1_test {}; balance {}; certainty {}"

    def _logging(
        self, train_metrics: List[float], test_metrics: List[float], epoch: int
    ) -> None:
        train_metrics = {
            met_name: met_val
            for met_name, met_val in zip(self.metric_names, train_metrics)
        }

        test_metrics = {
            met_name: met_val
            for met_name, met_val in zip(self.metric_names, test_metrics)
        }

        for mode, dict_metrics in zip(['train', 'test'], [train_metrics, test_metrics]):
            for metric in self.dict_logging[mode].keys():
                self.dict_logging[mode][metric].append(dict_metrics[metric])
                if self.logger:
                    self.logger.add_scalar(metric + "/" + mode, dict_metrics[metric], epoch)

        if epoch % self.print_every == 0:
            print_line = self.print_line.format(
                epoch + 1,
                round(train_metrics["loss"], 3),
                round(train_metrics["accuracy"], 3),
                round(test_metrics["loss"], 3),
                round(test_metrics["accuracy"], 3),
                round(test_metrics["f1"], 3),
                round(test_metrics["balance_pred"], 3),
                round(test_metrics["certainty"], 3),
            )
            print(print_line)

    def train_model(
        self, train_loader: DataLoader, valid_loader: DataLoader
    ) -> Dict[str, float]:
        if self.model.self_supervised and self.train_self_supervised:
            print("Training self-supervised model")
            X_train = train_loader.dataset.X.unsqueeze(-1).numpy()
            self.model.train_embedding(X_train, verbose=True)
            print("Training self-supervised part is finished")

        if self.early_stop_patience and self.early_stop_patience != "None":
            earl_stopper = EarlyStopper(self.early_stop_patience)

        self._init_logging(["loss"] + self.estimator.get_metrics_names())

        for epoch in range(self.n_epochs):
            train_metrics_epoch = self._run_epoch(train_loader, mode="train")
            test_metrics_epoch = self._run_epoch(valid_loader, mode="valid")

            self._logging(train_metrics_epoch, test_metrics_epoch, epoch)

            if self.early_stop_patience and self.early_stop_patience != "None":
                res_early_stop = earl_stopper.early_stop(test_metrics_epoch[0])
                if res_early_stop:
                    break

            if self.scheduler:
                self.scheduler.step()

        metrics_names = ['loss'] +self.estimator.get_metrics_names()
        test_metrics_epoch = {name: val for name, val in zip(metrics_names, test_metrics_epoch)}
        return test_metrics_epoch

    def _train_step(self, X: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor]:
        self.optimizer.zero_grad()

        y_preds = self.model(X)
        loss = self.criterion(y_preds, labels)

        loss.backward()
        self.optimizer.step()

        return loss, y_preds

    def _valid_step(self, X: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor]:
        with torch.no_grad():
            y_preds = self.model(X)
            loss = self.criterion(y_preds, labels)
        return loss, y_preds

    def _run_epoch(self, loader: DataLoader, mode: str = "train") -> List[float]:

        if mode not in ['train', 'valid']:
            raise ValueError("mode should be train or valid")
        losses = 0
        y_all_pred = torch.tensor([])
        y_all_pred_prob = torch.tensor([])
        y_all_true = torch.tensor([])

        self.model.train(mode=='train')
        for X, labels in loader:
            X = X.to(self.device)
            labels = labels.to(self.device)

            if mode == "train":
                loss, y_preds = self._train_step(X, labels)
            elif mode == "valid":
                loss, y_preds = self._valid_step(X, labels)

            if self.multiclass:
                y_pred = torch.argmax(y_preds, axis=1)
            else:
                y_pred = torch.round(y_preds)

            losses += loss
            y_all_true = torch.cat((y_all_true, labels.cpu().detach()), dim=0)
            y_all_pred_prob = torch.cat(
                (y_all_pred_prob, y_preds.cpu().detach()), dim=0
            )
            y_all_pred = torch.cat((y_all_pred, y_pred.cpu().detach()), dim=0)

        mean_loss = losses.cpu().detach().numpy() / len(loader)

        y_all_pred = y_all_pred.numpy().reshape([-1, 1])
        y_all_pred_prob = y_all_pred_prob.numpy().reshape([-1, 1])
        y_all_true = y_all_true.numpy().reshape([-1, 1])

        metrics = self.estimator.estimate(y_all_true, y_all_pred, y_all_pred_prob)

        metrics = [mean_loss] + metrics
        return metrics

    def save_metrics_as_csv(self, path: str) -> None:
        res = pd.DataFrame([])
        for split, metrics in self.dict_logging.items():
            df_metrics = pd.DataFrame(metrics)
            df_metrics["epoch"] = np.arange(1, len(df_metrics) + 1)
            df_metrics["split"] = split
            res = pd.concat([res, df_metrics])

        res = res.round(4)
        res.to_csv(path, index=False)

    def save_result(self, save_path: str, model_name: str, task: Task=None) -> None:

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        full_path = os.path.join(save_path,  model_name)
        torch.save(self.model.state_dict(), full_path  + ".pt")
        if task:
            task.upload_artifact(name='model_weights', artifact_object=full_path)
        self.save_metrics_as_csv(full_path + "_metrics.csv")


class DiscTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        attack: BaseIterativeAttack,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        attack_scheduler: AttackScheduler,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        multiclass: bool = False,
        train_self_supervised: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
            multiclass=multiclass,
            train_self_supervised=train_self_supervised,
        )

        self.train_attack = attack
        self.test_attack = copy.deepcopy(attack)
        self.attack_scheduler = attack_scheduler

    @staticmethod
    def initialize_with_params(
        model_name: str = "LSTM",
        model_params: Dict = None,
        attack_name: str = "FGSM",
        attack_params: Dict = None,
        criterion_name: str = "BCELoss",
        criterion_params: Dict = None,
        optimizer_name: str = "Adam",
        optimizer_params: Dict = None,
        scheduler_name: str = "None",
        scheduler_params: Dict = None,
        attack_scheduler_name: str = "None",
        attack_scheduler_params: Dict = None,
        n_epochs: int = 30,
        early_stop_patience: int = None,
        logger: Any = None,
        print_every: int = 5,
        device: str = "cpu",
        seed: int = 0,
        multiclass: bool = False,
        train_self_supervised: bool = False,
    ):
        fix_seed(seed)
        if model_params == "None" or not model_params:
            model_params = {}
        if criterion_params == "None" or not criterion_params:
            criterion_params = {}
        if optimizer_params == "None" or not optimizer_params:
            optimizer_params = {}
        if scheduler_params == "None" or not scheduler_params:
            scheduler_params = {}

        model = get_model(model_name, model_params, device=device)
        criterion = get_criterion(criterion_name, criterion_params)
        optimizer = get_optimizer(optimizer_name, model.parameters(), optimizer_params)
        scheduler = get_scheduler(scheduler_name, optimizer, scheduler_params)

        attack = get_attack(attack_name, attack_params)
        attack_scheduler = get_attack_scheduler(
            attack_scheduler_name, attack, attack_scheduler_params
        )

        return DiscTrainer(
            model=model,
            attack=attack,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            attack_scheduler=attack_scheduler,
            n_epochs=n_epochs,
            early_stop_patience=early_stop_patience,
            logger=logger,
            print_every=print_every,
            device=device,
            multiclass=multiclass,
            train_self_supervised=train_self_supervised,
        )

    @staticmethod
    def initialize_with_optimization(
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optuna_params: Dict,
        const_params: Dict,
        transform = None,
    ):
        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )

        study.optimize(
            partial(
                DiscTrainer.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                optim_metric=optuna_params["optim_metric"],
                const_params=const_params,
                train_loader=train_loader,
                valid_loader=valid_loader,
                transform=transform,
            ),
            n_trials=optuna_params["n_trials"],
        )

        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        print("DEFAULT", default_params)
        best_params = study.best_params.copy()
        print("BEST", best_params)
        best_params = update_dict_params(default_params, best_params)
        best_params = update_params_with_attack_params(const_params, best_params)
        print("Best parameters are - %s", best_params)
        return DiscTrainer.initialize_with_params(**best_params)

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        optim_metric: str,
        const_params: Dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        transform = None
    ) -> float:
        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)
        initial_model_parameters = update_params_with_attack_params(
            const_params, initial_model_parameters
        )

        model = DiscTrainer.initialize_with_params(**initial_model_parameters)
        last_epoch_metrics = model.train_model(train_loader, valid_loader, transform)
        return last_epoch_metrics[optim_metric]

    def _generate_adversarial_data(
        self, loader: DataLoader, transform=None, train=False
    ) -> DataLoader:
        X_orig = torch.tensor(loader.dataset.X)
        attack = self.train_attack if train else self.test_attack
        X_adv = attack.apply_attack(loader, self.logger).squeeze(-1)

        assert X_orig.shape == X_adv.shape

        disc_labels_zeros = torch.zeros_like(loader.dataset.y)
        disc_labels_ones = torch.ones_like(loader.dataset.y)

        new_x = torch.concat([X_orig, X_adv], dim=0)
        new_y = torch.concat([disc_labels_zeros, disc_labels_ones], dim=0)

        dataset_class = loader.dataset.__class__
        dataset = dataset_class(new_x, new_y, transform)

        loader = DataLoader(dataset, batch_size=loader.batch_size, shuffle=True)

        return loader

    def train_model(
        self, train_loader: DataLoader, valid_loader: DataLoader, transform=None, logger=None
    ) -> Dict[str, float]:

        if logger:
            self.logger = logger

        if self.model.self_supervised and self.train_self_supervised:
            print("Training self-supervised model")
            X_train = train_loader.dataset.X.unsqueeze(-1).numpy()
            self.model.train_embedding(X_train, verbose=True)
            print("Training self-supervised part is finished")

        if self.early_stop_patience and self.early_stop_patience != "None":
            earl_stopper = EarlyStopper(self.early_stop_patience)

        test_data_size = valid_loader.dataset.X.shape
        test_batch_size = valid_loader.batch_size
        # print()
        self.test_attack.update_data_batch_size(test_data_size, test_batch_size)

        adv_train_loader = self._generate_adversarial_data(train_loader, transform, train=True)
        adv_valid_loader = self._generate_adversarial_data(valid_loader, train=False)

        attack_sch_param_name = self.attack_scheduler.param_name
        prev_attack_sch_param = getattr(self.train_attack, attack_sch_param_name)

        self._init_logging(["loss"] + self.estimator.get_metrics_names() + [attack_sch_param_name])

        for epoch in range(self.n_epochs):
            train_metrics_epoch = self._run_epoch(adv_train_loader, mode="train")
            test_metrics_epoch = self._run_epoch(adv_valid_loader, mode="valid")

            train_metrics_epoch = list(train_metrics_epoch) + [prev_attack_sch_param]
            test_metrics_epoch = list(test_metrics_epoch) + [prev_attack_sch_param]

            self._logging(train_metrics_epoch, test_metrics_epoch, epoch)

            if self.early_stop_patience and self.early_stop_patience != "None":
                res_early_stop = earl_stopper.early_stop(test_metrics_epoch[0])
                if res_early_stop:
                    break

            if self.scheduler:
                self.scheduler.step()

            if self.attack_scheduler:
                self.train_attack = self.attack_scheduler.step()
                new_attack_sch_param = getattr(self.train_attack, self.attack_scheduler.param_name)
                setattr(self.test_attack, self.attack_scheduler.param_name, new_attack_sch_param)

                if prev_attack_sch_param != new_attack_sch_param and epoch + 1 != self.n_epochs:
                    if hasattr(self.train_attack, 'reinit_attack'):
                        self.train_attack.reinit_attack()
                        self.test_attack.reinit_attack()
                        
                    prev_attack_sch_param = new_attack_sch_param
                    print(f"----- New {attack_sch_param_name}", round(prev_attack_sch_param, 3))
                    adv_train_loader = self._generate_adversarial_data(
                        train_loader, transform, train=True
                    )
                    adv_valid_loader = self._generate_adversarial_data(valid_loader, train=False)

        metrics_names = ['loss'] +self.estimator.get_metrics_names()
        test_metrics_epoch = {name: val for name, val in zip(metrics_names, test_metrics_epoch)}
        return test_metrics_epoch

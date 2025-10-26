from abc import ABC, abstractmethod
from copy import copy
from functools import partial
from pyclbr import Class
from typing import Dict

import optuna
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from optuna.trial import Trial
from torch.utils.data import DataLoader

from src.utils import (
    collect_default_params,
    get_optimization_dict,
    update_dict_params,
    update_params_with_attack_params,
)


class BaseIterativeAttack(ABC):
    def __init__(
        self, model: torch.nn.Module, n_steps: int = 50, n_classes=2, *args, **kwargs
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.n_steps = n_steps
        self.n_classes = n_classes

    @abstractmethod
    def step(self, X: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def initialize_with_params(self, params) -> object:
        return self.__class__(**dict(params))

    def initialize_with_optimization(
        self,
        loader: DataLoader,
        optuna_params: Dict,
        const_params: Dict,
    ) -> object:
        attack_class = self.__class__

        study = optuna.create_study(
            direction="maximize",
            sampler=instantiate(optuna_params["sampler"]),
            pruner=instantiate(optuna_params["pruner"]),
        )
        study.optimize(
            partial(
                BaseIterativeAttack.objective,
                params_vary=optuna_params["hyperparameters_vary"],
                const_params=const_params,
                loader=loader,
                optim_metric=optuna_params["optim_metric"],
                attack_class=attack_class,
            ),
            n_trials=optuna_params["n_trials"],
        )

        default_params = collect_default_params(optuna_params["hyperparameters_vary"])
        best_params = study.best_params.copy()
        best_params = update_dict_params(default_params, best_params)
        best_params = update_params_with_attack_params(const_params, best_params)

        print("Best parameters are - %s", best_params)
        return attack_class(**dict(best_params))

    @staticmethod
    def objective(
        trial: Trial,
        params_vary: DictConfig,
        const_params: Dict,
        loader: DataLoader,
        attack_class: Class,
        optim_metric: str = "F_EFF_CONC",
    ) -> float:
        params = copy(const_params)

        initial_model_parameters, _ = get_optimization_dict(params_vary, trial)
        initial_model_parameters = dict(initial_model_parameters)
        params.update(initial_model_parameters)

        attack = attack_class(**dict(params))
        attack.apply_attack(loader)
        results = attack.get_metrics()
        last_step_metrics = results.iloc[-1]

        return last_step_metrics[optim_metric]
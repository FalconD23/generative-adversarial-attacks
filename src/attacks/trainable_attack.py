from typing import List

import torch

from src.attacks.base_attacks import BaseIterativeAttack
from src.attacks.procedures import TrainableBatchIterativeAttack
from src.estimation import BaseEstimator
from src.utils import req_grad 


class TrainAttack(BaseIterativeAttack, TrainableBatchIterativeAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        gen_model: torch.nn.Module,
        criterion: torch.nn.Module,
        estimator: BaseEstimator,
        logger=None,
        train_eps: float = 5,
        eps: float = 0.03,
        n_steps: int = 10,
        is_clamped: bool = False,
        n_classes = 2,
        *args,
        **kwargs,
    ) -> None:
        
        
        BaseIterativeAttack.__init__(self, model=model, n_steps=n_steps, n_classes=n_classes)
        TrainableBatchIterativeAttack.__init__(self, gen_model=gen_model, estimator=estimator, logger=logger, n_classes=n_classes)
        self.criterion = criterion
        self.is_clamped = is_clamped
        self.train_eps = train_eps
        self.eps = eps

        req_grad(self.model, False)
        req_grad(self.gen_model, True)
        self.gen_model.train()

        self.is_regularized = False
        self.n_classes = n_classes

        self.model

    def step(self, X: torch.Tensor, y_true: torch.Tensor, mode='val') -> torch.Tensor:
        if mode=='train':
            self.gen_model.train()
            delta_tilte = self.gen_model(X)
            delta_norm = (delta_tilte - delta_tilte.mean(dim=0)) / (delta_tilte.std() + 1e-5)
            delta_norm = torch.tanh(delta_tilte)
            delta = self.train_eps * delta_norm
            # print(torch.norm(delta), torch.norm(delta_norm), torch.norm(X))
        else:
            self.gen_model.eval()
            delta_tilte = self.gen_model(X)
            delta = self.eps * torch.sign(delta_tilte)
            # print(torch.norm(delta))

        if self.is_clamped:
            delta = torch.clamp(delta, -1, 1)
        
        X_adv = X + delta

        return X_adv

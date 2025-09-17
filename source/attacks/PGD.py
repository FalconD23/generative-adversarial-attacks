import torch
import torch.nn.functional as F
from source.attacks.base_attack import Attack


class PGDAttack(Attack):
    def __init__(self, eps, n_iter=40, alpha=None, clamp=None,
                 rand_init=True, targeted=False, equal_eps=True):
        """
        PGD (L∞): многошаговый FGSM с проекцией в L∞-шар.
        - eps: радиус шара
        - n_iter: число итераций
        - alpha: шаг; по умолчанию 1.5*eps/n_iter
        - clamp: (min, max) диапазон данных или None
        - rand_init: случайная инициализация внутри шара
        - targeted: таргетированная атака (минимизируем loss целевого y)
        - equal_eps: требовать ||delta||_inf = eps (по возможности)
        """
        super().__init__(eps, clamp)
        self.n_iter = n_iter
        self.alpha = alpha or 1.5 * eps / n_iter
        self.rand_init = rand_init
        self.targeted = targeted
        self.equal_eps = equal_eps

    @staticmethod
    def _project_linf(delta, eps, equal_eps: bool):
        # проекция в L∞-шар
        delta = torch.clamp(delta, min=-eps, max=eps)
        if not equal_eps:
            return delta
        # дотягиваем до сферы: max(|delta|) == eps (если возможно)
        flat = delta.detach().abs().flatten(1)  # (N, ...)
        amax = flat.amax(dim=1)  # (N,)
        scale = (eps / (amax + 1e-12)).view(-1, *[1] * (delta.dim() - 1))
        delta = (delta * scale).clamp(-eps, eps)
        return delta

    @staticmethod
    @torch.no_grad()
    def _rand_init_linf(x, eps, clamp, equal_eps: bool):
        delta = torch.empty_like(x).uniform_(-eps, eps)
        if equal_eps:
            flat = delta.abs().flatten(1)
            amax = flat.amax(dim=1)
            scale = (eps / (amax + 1e-12)).view(-1, *[1] * (delta.dim() - 1))
            delta = (delta * scale).clamp(-eps, eps)
        x0 = x + delta
        if clamp is not None:
            x0 = torch.clamp(x0, *clamp)  # может «урезать» норму ниже eps, это физ. ограничение
        return x0

    def __call__(self, model, x, y):
        x = x.detach()
        # инициализация
        if self.rand_init:
            x_adv = self._rand_init_linf(x, self.eps, self.clamp, self.equal_eps)
        else:
            x_adv = x.clone()

        for _ in range(self.n_iter):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            if self.targeted:
                loss = -loss

            model.zero_grad(set_to_none=True)
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            with torch.no_grad():
                # шаг по знаку градиента (L∞)
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # проекция в L∞-шар (и опционально на сферу)
                delta = self._project_linf(x_adv - x, self.eps, self.equal_eps)
                x_adv = x + delta
                if self.clamp is not None:
                    x_adv = torch.clamp(x_adv, *self.clamp)
            x_adv = x_adv.detach()

        return x_adv

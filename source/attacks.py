import torch
import torch.nn.functional as F


class Attack:
    def __init__(self, eps: float, clamp: tuple[float | None] | None = None):
        self.eps = eps
        self.clamp = clamp


class FGSMAttack(Attack):
    def __call__(self, model, x, y):
        x_req = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_req), y)
        loss.backward()
        delta = self.eps * x_req.grad.sign()
        x_adv = x + delta
        if self.clamp is not None:
            x_adv = torch.clamp(x_adv, *self.clamp)
        return x_adv.detach()


class iFGSMAttack(Attack):
    def __init__(self, eps, n_iter=20, alpha=None, clamp=None,
                 rand_init=True, momentum=0.9):
        super().__init__(eps, clamp)
        self.n_iter, self.alpha = n_iter, alpha or 1.5 * eps / n_iter
        self.rand_init, self.mu = rand_init, momentum

    def __call__(self, model, x, y):
        x_adv = x.detach()
        if self.rand_init:
            x_adv = x_adv + torch.empty_like(x).uniform_(-self.eps, self.eps)
            if self.clamp is not None:
                x_adv = torch.clamp(x_adv, *self.clamp)
        g = torch.zeros_like(x)
        for _ in range(self.n_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            model.zero_grad()
            loss.backward()
            grad = x_adv.grad / x_adv.grad.abs().mean(dim=(1, 2), keepdim=True)
            g = self.mu * g + grad
            x_adv = x_adv + self.alpha * g.sign()
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = x + delta
            if self.clamp is not None:
                x_adv = torch.clamp(x_adv, *self.clamp)
            x_adv = x_adv.detach()
        return x_adv


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


class ModelBasedAttack(Attack):
    def __init__(self, attacker, eps, clamp=None, is_iter=False):
        super().__init__(eps, clamp)
        self.attacker = attacker.eval()
        self.is_iter = is_iter

    @torch.no_grad()
    def __call__(self, model, x, y):
        if not self.is_iter:
            # delta = self.eps * self.attacker(x).sign()
            delta = self.eps * torch.tanh(self.attacker(x))
            # print(f"L2 norm = {torch.norm(delta, p=2)}, L_infty norm = {torch.norm(delta, p=float('inf'))}")
            x_adv = x + delta
            if not self.clamp is None:
                x_adv = torch.clamp(x_adv, *self.clamp)
            return x_adv
        else:
            return self.attacker(x)

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


class ModelBasedAttack(Attack):
    def __init__(self, attacker, eps, clamp=None):
        super().__init__(eps, clamp)
        self.attacker = attacker.eval()

    @torch.no_grad()
    def __call__(self, model, x, y):
        delta = self.eps * torch.tanh(self.attacker(x))
        print(f"L2 norm = {torch.norm(delta, p=2):.3f}, L_infty norm = {torch.norm(delta, p=float('inf')):.3f}")
        x_adv = x + delta
        if self.clamp is not None:
            x_adv = torch.clamp(x_adv, *self.clamp)
        return x_adv


import torch
import torch.nn.functional as F
from source.attacks.base_attack import Attack

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
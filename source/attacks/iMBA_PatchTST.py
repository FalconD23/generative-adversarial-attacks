import torch
import torch.nn as nn


class _STClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        return x.clamp(lo, hi)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def st_clamp(x, lo, hi, use_st: bool):
    return _STClamp.apply(x, lo, hi) if use_st else x.clamp(lo, hi)


def _normalize_step(d, kind: str = None):
    if kind in (None, "none"):
        return d
    if kind == "meanabs":
        s = d.abs().mean(dim=tuple(range(1, d.dim())), keepdim=True) + 1e-12
        return d / s
    if kind == "l2":
        s = torch.norm(d.flatten(1), p=2, dim=1).view(-1, *[1] * (d.dim() - 1)) + 1e-12
        return d / s
    if kind == "linf":
        s = d.abs().flatten(1).amax(dim=1).view(-1, *[1] * (d.dim() - 1)) + 1e-12
        return d / s
    raise ValueError(f"Unknown normalize kind: {kind}")


class IterModelAttack(nn.Module):
    """
    x_{t+1} = Proj( Clamp( x_t + alpha * dir(attacker(x_t), momentum) ) ).
    Флаги по умолчанию повторяют твоё текущее поведение: БЕЗ проекции/клампа/моментума.
    """

    def __init__(
            self,
            attacker: nn.Module,
            eps: float,
            n_iter: int = 10,
            alpha = None,
            # --- сохранённые аргументы для совместимости ---
            clamp = None,  # было data_clamp в train-функции
            rand_init: bool = True,
            use_sign: bool = False,
            equal_eps: bool = False,
            bpda: bool = True,
            # --- новые флаги (все выключены по умолчанию) ---
            proj: str = "none",  # "none" | "linf" | "l2"
            proj_equal_eps: bool = False,  # дотягивать до ||δ|| = eps
            data_clamp = None,  # (min, max) по данным; по умолчанию None
            momentum_mu: float = 0.0,  # 0.0 = без momentum (как у тебя)
            step_normalize =  None,  # "meanabs" | "l2" | "linf" | None
            step_noise_std: float = 0.0,  # гаусс. шум в шаге (для стабилизации)
    ):
        super().__init__()
        self.attacker = attacker
        self.eps = eps
        self.n_iter = n_iter
        self.alpha = alpha or (1.0 * eps / max(1, n_iter))
        # совместимость:
        self.rand_init = rand_init
        self.use_sign = use_sign
        self.equal_eps = equal_eps
        self.bpda = bpda
        # новые опции:
        self.proj = proj
        self.proj_equal_eps = proj_equal_eps
        self.data_clamp = data_clamp if clamp is None else clamp  # поддержка старого имени
        self.momentum_mu = momentum_mu
        self.step_normalize = step_normalize
        self.step_noise_std = step_noise_std

    @staticmethod
    def _project(delta, eps, mode: str, equal_eps: bool, bpda: bool):
        if mode == "none":
            return delta
        if mode == "linf":
            delta = st_clamp(delta, -eps, eps, use_st=bpda)
            if equal_eps:
                amax = delta.detach().abs().flatten(1).amax(dim=1)  # (B,)
                scale = (eps / (amax + 1e-12)).view(-1, *[1] * (delta.dim() - 1))
                delta = delta * scale
                delta = st_clamp(delta, -eps, eps, use_st=bpda)
            return delta
        if mode == "l2":
            flat = delta.flatten(1)
            nrm = torch.norm(flat, p=2, dim=1).view(-1, *[1] * (delta.dim() - 1)) + 1e-12
            # проекция в L2-шар
            delta = delta * torch.clamp(eps / nrm, max=1.0)
            if equal_eps:
                delta = delta * (eps / (nrm + 1e-12))
            return delta
        raise ValueError(f"Unknown projection mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        # random start
        if self.rand_init:
            if self.proj == "linf":
                delta0 = torch.empty_like(x0).uniform_(-self.eps, self.eps)
            elif self.proj == "l2":
                delta0 = torch.randn_like(x0)
                delta0 = self._project(delta0, self.eps, "l2", False, self.bpda)
            else:  # none
                delta0 = torch.empty_like(x0).uniform_(-self.eps, self.eps)
            x_adv = x0 + delta0
        else:
            x_adv = x0.clone()

        g = torch.zeros_like(x_adv)  # momentum buffer

        for _ in range(self.n_iter):
            step_dir = self.attacker(x_adv)
            step_dir = step_dir.sign() if self.use_sign else torch.tanh(step_dir)
            if self.step_normalize:
                step_dir = _normalize_step(step_dir, self.step_normalize)
            if self.step_noise_std > 0:
                step_dir = step_dir + self.step_noise_std * torch.randn_like(step_dir)

            # momentum (как в MI-FGSM: накопление направления)
            if self.momentum_mu > 0.0:
                g = self.momentum_mu * g + step_dir
                step_dir = g.sign() if self.use_sign else g

            x_adv = x_adv + self.alpha * step_dir

            # проекция в ε-шар (если включена)
            delta = x_adv - x0
            delta = self._project(delta, self.eps, self.proj, self.proj_equal_eps, self.bpda)
            x_adv = x0 + delta

            # clamp по данным (если указан)
            if self.data_clamp is not None:
                x_adv = st_clamp(x_adv, self.data_clamp[0], self.data_clamp[1], use_st=self.bpda)

        return x_adv

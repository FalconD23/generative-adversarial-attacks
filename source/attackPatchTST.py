from typing import Optional

import torch
from torch import nn

from activation import Activation


class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.self_supervised = False


class PatchTST(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = mdls.PatchTST(**kwargs).float()
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)
        return self.final_activation(output).squeeze(-1)


class AttackPatchTST(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            x_dim: int = 1,
            activation_type: str = "tanh",
            patch_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        patch_kwargs = patch_kwargs or {}
        # гарантируем нужный выход PatchTST
        patch_kwargs.update(dict(c_in=x_dim,
                                 c_out=hidden_dim,
                                 pred_dim=hidden_dim))

        self.step_model = PatchTST(activation_type="identity", **patch_kwargs)
        self.fc = nn.Linear(hidden_dim, x_dim)
        self.act = Activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.step_model(x)  # (B, 1, 1, hidden_dim)

        h = h.view(B, -1)
        # либо h = h.squeeze(1).squeeze(1)

        h = h.unsqueeze(1).expand(-1, L, -1)  # (B, L, hidden_dim)
        return self.fc(self.act(h))  # (B, L, C)

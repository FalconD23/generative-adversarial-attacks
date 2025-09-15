import torch
import torch.nn as nn
from typing import Optional
from source.models.PatchTST_clf import PatchTST
from source.models.activation import Activation

class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        x_dim: int = 1,
        activation_type: str = "identity",
        patch_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.x_dim = x_dim
        patch_kwargs = dict(patch_kwargs or {})


        patch_kwargs.update(dict(
            c_in=x_dim,
            c_out=n_classes,
            # pred_dim=n_classes,
        ))

        self.body = PatchTST(activation_type="identity", **patch_kwargs)
        self.fin = Activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.ndim == 3 and x.shape[1] != self.x_dim:   # (B,L,C)
        #     x = x.transpose(1, 2)
        return self.fin(self.body(x))

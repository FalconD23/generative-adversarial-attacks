from typing import Optional
import torch
from torch import nn
import tsai.models.all as mdls

from .base_model import BaseModel
from .utils import Activation


class PatchTST(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", c_in=1, **kwargs) -> None:
        super().__init__()
        self.model = mdls.PatchTST(c_in=c_in, **kwargs).float()
        self.final_activation = Activation(activation_type)
        self.c_in = c_in

        if self.c_in > 1:
            self.pred_head = torch.nn.Sequential(
                torch.nn.Linear(self.c_in, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
                )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)

        if self.c_in > 1:
            output = self.pred_head(output.transpose(2, 1))

        return self.final_activation(output).squeeze(-1)
    


class GenPatchTST(BaseModel):
    """
    PatchTST-based surrogate model for generating adversarial perturbations.
    
    Architecture: Input -> PatchTST -> Activation -> Linear -> Output
    """
    
    def __init__(
        self,
        seq_len: int = 200,
        hidden_dim: int = 128,
        c_in: int = 1,
        activation_type: str = "tanh",
        patch_kwargs: dict = {}
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.c_in = c_in
        self.seq_len = seq_len
        self.activation_type = activation_type
        
        # PatchTST backbone
        patch_kwargs = dict(patch_kwargs) if patch_kwargs else {}
        # Ensure correct output dimensions for PatchTST
        patch_kwargs.update(dict(
            c_in=c_in,
            c_out=hidden_dim,
            pred_dim=hidden_dim,
            seq_len = seq_len
        ))
        
        self.step_model = mdls.PatchTST(**patch_kwargs)
        self.fc = nn.Linear(hidden_dim, c_in)
        self.act = Activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, x_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, x_dim)
        """
        B, L, _ = x.shape
        # PatchTST expects (B, C, L) format
        x_transposed = x.transpose(1, 2)  # (B, x_dim, L)
        
        # Get hidden representation: (B, hidden_dim, 1)
        h = self.step_model(x_transposed)
        
        # Reshape: (B, hidden_dim, 1) -> (B, hidden_dim)
        h = h.view(B, -1)
        # Expand to sequence length: (B, hidden_dim) -> (B, L, hidden_dim)
        #! CONST SHIFT PROBLEM
        h = h.unsqueeze(1).expand(-1, L, -1)
        return self.fc(self.act(h))


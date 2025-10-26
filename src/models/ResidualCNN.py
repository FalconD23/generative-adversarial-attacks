import torch
from torch import nn
from tsai.models.all import ResCNN

from .base_model import BaseModel
from .utils import Activation


class ResidualCNN(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = ResCNN(**kwargs).float()
        self.final_activation = Activation(activation_type)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        output = self.model(X)
        return self.final_activation(output)
    

class GenResidualCNN(BaseModel):
    """
    Residual CNN-based surrogate model for generating adversarial perturbations.
    
    Architecture: Input -> ResCNN -> Activation -> Linear -> Output
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        x_dim: int = 1,
        activation_type: str = 'tanh',
        rescnn_kwargs: dict = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.x_dim = x_dim
        self.activation_type = activation_type
        
        # ResCNN backbone
        rescnn_kwargs = rescnn_kwargs or {}
        self.step_cnn = ResCNN(
            c_in=x_dim, 
            c_out=hidden_dim, 
            **rescnn_kwargs
        )
        
        # Final linear layer
        self.fc = nn.Linear(hidden_dim, x_dim)
        
        # Activation layer
        self.act = Activation(activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, x_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, x_dim)
        """
        B, L, C = x.shape
        # Reshape for ResCNN: (B*L, C, 1) -> (B*L, hidden_dim, 1)
        x_flat = x.contiguous().view(B * L, C, 1)
        h = self.step_cnn(x_flat)  # (B*L, hidden_dim, 1)
        
        # Reshape back: (B*L, hidden_dim, 1) -> (B, L, hidden_dim)
        h = h.view(B, L, -1)
        return self.fc(self.act(h))

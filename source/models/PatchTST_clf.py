import torch
import tsai.models.all as mdls
from source.models.activation import Activation

class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.self_supervised = False

# class PatchTST(BaseModel):
#     def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
#         super().__init__()
#         self.model = mdls.PatchTST(**kwargs).float()
#         self.final_activation = Activation(activation_type)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         X = X.transpose(1, 2)
#         output = self.model(X)
#         return self.final_activation(output).squeeze(-1)


class PatchTST(BaseModel):
    def __init__(self, activation_type: str = "sigmoid", **kwargs) -> None:
        super().__init__()
        self.model = mdls.PatchTST(**kwargs).float()
        self.final_activation = Activation(activation_type)
        self.kwargs = kwargs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # ожидаем (B, L, C) → переводим в (B, C, L)
        if X.ndim == 3 and X.shape[1] != self.kwargs.get('c_in', 1):
            X = X.transpose(1, 2)

        output = self.model(X)  # форма зависит от настроек PatchTST

        # Приводим к (B, C): сворачиваем временную ось разумно
        if output.ndim == 3:
            # варианты: (B, C, T) или (B, T, C)
            # if output.shape[-1] == 1:
            #     output = output.squeeze(-1)          # (B, C, 1) -> (B, C)
            if output.shape[1] == 1:
                output = output.squeeze(1)           # (B, 1, C) -> (B, C)


        return self.final_activation(output)  # (B, C)

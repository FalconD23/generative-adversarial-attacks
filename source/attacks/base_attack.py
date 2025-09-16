import torch
import torch.nn.functional as F
from typing import Tuple

class Attack:
    def __init__(self, eps: float, clamp: Tuple[float]=None):
        self.eps = eps
        self.clamp = clamp
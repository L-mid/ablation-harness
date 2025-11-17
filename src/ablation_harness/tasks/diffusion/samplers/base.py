# samplers/base.py
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Abstract protocall method for defining samplers"""

    def __init__(self, q, *, nfe: int, eta: float = 0.0, device: str = "cpu"):
        self.q, self.nfe, self.eta, self.device = q, nfe, eta, device

    @abstractmethod
    def step(self, model, x_t, t, t_prev):  # t_prev is ignored by DDPM
        ...

    @abstractmethod
    def sample(self, model, shape, seed=0): ...

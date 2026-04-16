from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias

import torch
from tensordict.nn import TensorDictModule
from torch import nn

TdKey: TypeAlias = str | tuple[str, ...]


class ModuleMixin(nn.Module):
    def create_module(self) -> TensorDictModule:
        raise NotImplementedError()


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


@dataclass
class MLPParams:
    input_dim: Optional[int]
    output_dim: int
    hidden_dims: list[int]


class MLPBlock(nn.Module):
    def __init__(self, params: MLPParams):
        super().__init__()
        self.params = params

        layers: list[nn.Module] = []
        in_dim = self.params.input_dim

        if not self.params.hidden_dims:
            layers.append(
                nn.LazyLinear(self.params.output_dim)
                if in_dim is None
                else nn.Linear(in_dim, self.params.output_dim)
            )
        else:
            for hidden_dim in self.params.hidden_dims:
                layers.append(
                    nn.LazyLinear(hidden_dim)
                    if in_dim is None
                    else nn.Linear(in_dim, hidden_dim)
                )
                layers.append(nn.SiLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, self.params.output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

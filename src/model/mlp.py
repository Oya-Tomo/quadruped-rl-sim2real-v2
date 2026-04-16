from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torch.distributions import Normal
from torchrl.data import TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import IndependentNormal

from src.model.base import MLPBlock, MLPParams, ModuleMixin, TdKey


@dataclass
class ActorParams:
    state_dim: Optional[int]
    action_dim: int
    hidden_dims: list[int]
    state_keys: list[TdKey]
    action_key: TdKey
    init_log_std: float = 0.0
    min_log_std: float = -4.0
    max_log_std: float = 2.0
    loc: float | torch.Tensor = 0.0
    scale: float | torch.Tensor = 1.0
    state_independent_std: bool = True


class ActorNetwork(ModuleMixin):
    def __init__(self, params: ActorParams):
        super().__init__()
        self.params = params
        if self.params.state_independent_std:
            self.mlp = MLPBlock(
                MLPParams(
                    input_dim=params.state_dim,
                    output_dim=params.action_dim,
                    hidden_dims=params.hidden_dims,
                )
            )
            self.log_std = nn.Parameter(
                torch.full(
                    (params.action_dim,), params.init_log_std, dtype=torch.float32
                )
            )
        else:
            self.mlp = MLPBlock(
                MLPParams(
                    input_dim=params.state_dim,
                    output_dim=params.action_dim * 2,
                    hidden_dims=params.hidden_dims,
                )
            )

    @property
    def config(self) -> dict:
        return asdict(self.params)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(state)
        if self.params.state_independent_std:
            mean = x
            log_std = self.log_std.expand_as(mean)
        else:
            mean, log_std = torch.chunk(x, 2, dim=-1)
        mean = mean + self.params.loc
        log_std = log_std.clamp(self.params.min_log_std, self.params.max_log_std)
        std = log_std.exp() * self.params.scale
        return mean, std

    def create_module(self) -> TensorDictModule:
        actor_out_keys = [self.params.action_key, "mean", "std"]
        return TensorDictSequential(
            TensorDictSequential(
                TensorDictModule(
                    lambda *x: torch.cat(x, dim=-1),
                    in_keys=self.params.state_keys,
                    out_keys=["_state"],
                ),
                TensorDictModule(self, in_keys=["_state"], out_keys=["mean", "std"]),
                selected_out_keys=["mean", "std"],
            ),
            TensorDictModule(
                lambda mean, std: Normal(mean, std).rsample(),
                in_keys=["mean", "std"],
                out_keys=[self.params.action_key],
            ),
            selected_out_keys=actor_out_keys,
        )

    def create_prob_actor_module(self, action_spec: TensorSpec) -> ProbabilisticActor:
        return ProbabilisticActor(
            module=TensorDictSequential(
                TensorDictModule(
                    lambda *x: torch.cat(x, dim=-1),
                    in_keys=self.params.state_keys,
                    out_keys=["_state"],
                ),
                TensorDictModule(self, in_keys=["_state"], out_keys=["loc", "scale"]),
                selected_out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            out_keys=[self.params.action_key],
            spec=action_spec,
            distribution_class=IndependentNormal,
            return_log_prob=True,
        )


@dataclass
class CriticParams:
    state_dim: Optional[int]
    hidden_dims: list[int]
    value_dim: int
    cost_dim: int | None
    state_keys: list[TdKey]
    value_key: TdKey
    value_cost_key: TdKey | None


class CriticNetwork(ModuleMixin):
    def __init__(self, params: CriticParams):
        super().__init__()
        self.params = params
        output_dim = params.value_dim + (params.cost_dim or 0)
        self.mlp = MLPBlock(
            MLPParams(
                input_dim=params.state_dim,
                output_dim=output_dim,
                hidden_dims=params.hidden_dims,
            )
        )

    @property
    def config(self) -> dict:
        return asdict(self.params)

    def forward(self, x: torch.Tensor):
        output = self.mlp(x)
        if self.params.cost_dim is not None and self.params.cost_dim > 0:
            value = output[..., : self.params.value_dim]
            costs = output[..., self.params.value_dim :]
            return value, costs
        return output

    def create_module(self) -> TensorDictModule:
        out_keys_list = (
            [self.params.value_key, self.params.value_cost_key]
            if self.params.cost_dim is not None and self.params.cost_dim > 0
            else [self.params.value_key]
        )
        return TensorDictSequential(
            TensorDictModule(
                lambda *x: torch.cat(x, dim=-1),
                in_keys=self.params.state_keys,
                out_keys=["_state"],
            ),
            TensorDictModule(self, in_keys=["_state"], out_keys=out_keys_list),
            selected_out_keys=out_keys_list,
        )


@dataclass
class HistoryEncoderParams:
    history_dim: Optional[int]
    hidden_dims: list[int]
    latent_dim: int
    history_keys: list[TdKey]
    latent_key: TdKey


class HistoryEncoderNetwork(ModuleMixin):
    def __init__(self, params: HistoryEncoderParams):
        super().__init__()
        self.params = params
        self.mlp = MLPBlock(
            MLPParams(
                input_dim=params.history_dim,
                output_dim=params.latent_dim,
                hidden_dims=params.hidden_dims,
            )
        )

    @property
    def config(self) -> dict:
        return asdict(self.params)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.mlp(history)

    def create_module(self) -> TensorDictModule:
        out_keys_list = [self.params.latent_key]
        return TensorDictSequential(
            TensorDictModule(
                lambda *x: torch.cat(x, dim=-1).flatten(start_dim=-2),
                in_keys=self.params.history_keys,
                out_keys=["_history"],
            ),
            TensorDictModule(self, in_keys=["_history"], out_keys=out_keys_list),
            selected_out_keys=out_keys_list,
        )

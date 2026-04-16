from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torch.distributions import Normal
from torchrl.data import Composite, UnboundedContinuous
from torchrl.envs import TensorDictPrimer

from src.model.base import MLPBlock, MLPParams, ModuleMixin, TdKey, module_device


@dataclass
class RNNCellParams:
    input_dim: int
    hidden_state_dim: int


class RNNCellBlock(nn.Module):
    def __init__(self, params: RNNCellParams):
        super().__init__()
        self.params = params
        self.rnn = nn.RNNCell(
            input_size=self.params.input_dim,
            hidden_size=self.params.hidden_state_dim,
        )

    def forward(self, rnn_input: torch.Tensor, hidden_state: torch.Tensor):
        hidden_n = self.rnn(rnn_input, hidden_state)
        return hidden_n, hidden_n


@dataclass
class ActorParams:
    state_dim: int
    action_dim: int
    hidden_dims: list[int]
    hidden_state_dim: int
    state_keys: list[TdKey]
    action_key: TdKey
    hidden_state_key: TdKey
    reset_hidden_key: Optional[TdKey]
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
        self.rnn_cell = RNNCellBlock(
            RNNCellParams(
                input_dim=self.params.state_dim,
                hidden_state_dim=self.params.hidden_state_dim,
            )
        )

        if self.params.state_independent_std:
            self.output_layer = MLPBlock(
                MLPParams(
                    input_dim=self.params.state_dim + self.params.hidden_state_dim,
                    output_dim=self.params.action_dim,
                    hidden_dims=self.params.hidden_dims,
                )
            )
            self.log_std = nn.Parameter(
                torch.full(
                    (params.action_dim,), params.init_log_std, dtype=torch.float32
                )
            )
        else:
            self.output_layer = MLPBlock(
                MLPParams(
                    input_dim=self.params.state_dim + self.params.hidden_state_dim,
                    output_dim=self.params.action_dim * 2,
                    hidden_dims=self.params.hidden_dims,
                )
            )

    @property
    def config(self) -> dict:
        return asdict(self.params)

    def make_tensordict_primer(self) -> TensorDictPrimer:
        return TensorDictPrimer(
            Composite(
                {
                    self.params.hidden_state_key: UnboundedContinuous(
                        (self.params.hidden_state_dim,),
                        device=module_device(self),
                        dtype=torch.float32,
                    )
                }
            ),
            expand_specs=True,
        )

    def forward(self, state: torch.Tensor, hidden_state: torch.Tensor):
        rnn_out, hidden_n = self.rnn_cell(state, hidden_state)
        mlp_input = torch.cat([state, rnn_out], dim=-1)
        x = self.output_layer(mlp_input)
        if self.params.state_independent_std:
            mean = x
            log_std = self.log_std.expand_as(mean)
        else:
            mean, log_std = torch.chunk(x, 2, dim=-1)
        mean = mean + self.params.loc
        log_std = log_std.clamp(self.params.min_log_std, self.params.max_log_std)
        std = log_std.exp() * self.params.scale
        return mean, std, hidden_n

    def create_module(self) -> TensorDictModule:
        out_keys_list = [
            self.params.action_key,
            "mean",
            "std",
            ("next", self.params.hidden_state_key),
        ]
        reset_hidden_module = (
            TensorDictModule(
                lambda hidden_state, reset_hidden: hidden_state
                * reset_hidden.logical_not().to(torch.float32),
                in_keys=[self.params.hidden_state_key, self.params.reset_hidden_key],
                out_keys=[self.params.hidden_state_key],
            )
            if self.params.reset_hidden_key is not None
            else TensorDictModule(
                lambda hidden_state: hidden_state,
                in_keys=[self.params.hidden_state_key],
                out_keys=[self.params.hidden_state_key],
            )
        )

        return TensorDictSequential(
            TensorDictSequential(
                reset_hidden_module,
                TensorDictModule(
                    lambda *x: torch.cat(x, dim=-1),
                    in_keys=self.params.state_keys,
                    out_keys=["_state"],
                ),
                TensorDictModule(
                    self,
                    in_keys=["_state", self.params.hidden_state_key],
                    out_keys=["mean", "std", ("next", self.params.hidden_state_key)],
                ),
                selected_out_keys=[
                    "mean",
                    "std",
                    ("next", self.params.hidden_state_key),
                ],
            ),
            TensorDictModule(
                lambda mean, std: Normal(mean, std).rsample(),
                in_keys=["mean", "std"],
                out_keys=[self.params.action_key],
            ),
            selected_out_keys=out_keys_list,
        )


@dataclass
class CriticParams:
    state_dim: int
    hidden_dims: list[int]
    hidden_state_dim: int
    value_dim: int
    cost_dim: int | None
    state_keys: list[TdKey]
    value_key: TdKey
    value_cost_key: TdKey | None
    hidden_state_key: TdKey
    reset_hidden_key: Optional[TdKey]


class CriticNetwork(ModuleMixin):
    def __init__(self, params: CriticParams):
        super().__init__()
        self.params = params
        self.rnn_cell = RNNCellBlock(
            RNNCellParams(
                input_dim=self.params.state_dim,
                hidden_state_dim=self.params.hidden_state_dim,
            )
        )
        output_dim = self.params.value_dim + (self.params.cost_dim or 0)
        self.output_layer = MLPBlock(
            MLPParams(
                input_dim=self.params.state_dim + self.params.hidden_state_dim,
                output_dim=output_dim,
                hidden_dims=self.params.hidden_dims,
            )
        )

    @property
    def config(self) -> dict:
        return asdict(self.params)

    def make_tensordict_primer(self) -> TensorDictPrimer:
        return TensorDictPrimer(
            Composite(
                {
                    self.params.hidden_state_key: UnboundedContinuous(
                        (self.params.hidden_state_dim,),
                        device=module_device(self),
                        dtype=torch.float32,
                    )
                }
            ),
            expand_specs=True,
        )

    def forward(self, state: torch.Tensor, hidden_state: torch.Tensor):
        rnn_out, hidden_n = self.rnn_cell(state, hidden_state)
        mlp_input = torch.cat([state, rnn_out], dim=-1)
        output = self.output_layer(mlp_input)
        if self.params.cost_dim is not None and self.params.cost_dim > 0:
            value = output[..., : self.params.value_dim]
            costs = output[..., self.params.value_dim :]
            return value, costs, hidden_n
        return output, hidden_n

    def create_module(self) -> TensorDictModule:
        reset_hidden_module = (
            TensorDictModule(
                lambda hidden_state, reset_hidden: hidden_state
                * reset_hidden.logical_not().to(torch.float32),
                in_keys=[self.params.hidden_state_key, self.params.reset_hidden_key],
                out_keys=[self.params.hidden_state_key],
            )
            if self.params.reset_hidden_key is not None
            else TensorDictModule(
                lambda hidden_state: hidden_state,
                in_keys=[self.params.hidden_state_key],
                out_keys=[self.params.hidden_state_key],
            )
        )

        out_keys_list = (
            [
                self.params.value_key,
                self.params.value_cost_key,
                ("next", self.params.hidden_state_key),
            ]
            if self.params.cost_dim is not None and self.params.cost_dim > 0
            else [self.params.value_key, ("next", self.params.hidden_state_key)]
        )
        return TensorDictSequential(
            TensorDictSequential(
                reset_hidden_module,
                TensorDictModule(
                    lambda *x: torch.cat(x, dim=-1),
                    in_keys=self.params.state_keys,
                    out_keys=["_state"],
                ),
                TensorDictModule(
                    self,
                    in_keys=["_state", self.params.hidden_state_key],
                    out_keys=out_keys_list,
                ),
                selected_out_keys=out_keys_list,
            ),
            selected_out_keys=out_keys_list,
        )

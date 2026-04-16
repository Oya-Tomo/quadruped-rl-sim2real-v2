from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

import tensordict
import torch
from tensordict import NestedKey, TensorDict
from torch.distributions import Normal, kl_divergence
from torchrl.collectors import Collector
from torchrl.envs import EnvBase
from torchrl.record.loggers import Logger

from src.algorithm.rms import RunningMeanStd
from src.algorithm.utils import sample_batch, sample_seq_batch
from src.model.base import ModuleMixin


@dataclass
class PPOParams:
    task_name: str
    loops: int
    steps_per_batch: int
    epochs_per_batch: int
    sub_batch_size: int
    checkpoint_interval: int
    checkpoint_dir: str
    gamma: float
    lam: float
    clip_epsilon: float
    value_clip_epsilon: float
    entropy_loss_coeff: float
    critic_loss_coeff: float
    value_loss_type: Literal["smooth_l1", "l1", "l2"]
    use_rms: bool
    max_grad_norm: float
    upper_kl: float
    desired_kl: float
    actor_lr: float
    actor_lr_max: float
    actor_lr_min: float
    critic_lr: float
    weight_decay: float
    normalize_advantages: bool
    normalize_advantages_type: Optional[Literal["standard", "std_only", "rms"]] = None
    action_key: NestedKey = "action"
    mean_key: NestedKey = "mean"
    std_key: NestedKey = "std"
    reward_key: NestedKey = "reward"
    value_key: NestedKey = "value"
    return_key: NestedKey = "return"
    advantage_key: NestedKey = "advantage"
    reset_key: NestedKey = "done"
    terminate_key: NestedKey = "terminated"
    info_keys: NestedKey = ("next", "info")
    rnn_mode: bool = False
    tbptt_steps: Optional[int] = None


class PPOTrainer:
    def __init__(
        self,
        params: PPOParams,
        env: EnvBase,
        actor_network: ModuleMixin,
        critic_network: ModuleMixin,
        logger: Optional[Logger] = None,
    ):
        self.params = params
        self.logger = logger

        self.env = env
        self.n_envs = env.batch_size[0]
        self.device = env.device

        self.actor_network = actor_network
        self.actor_module = self.actor_network.create_module()
        self.actor_optimizer = torch.optim.Adam(
            self.actor_network.parameters(),
            lr=self.params.actor_lr,
            weight_decay=self.params.weight_decay,
        )
        self.actor_lr = self.params.actor_lr

        self.critic_network = critic_network
        self.critic_module = self.critic_network.create_module()
        self.critic_optimizer = torch.optim.Adam(
            self.critic_network.parameters(),
            lr=self.params.critic_lr,
            weight_decay=self.params.weight_decay,
        )
        self.rms = (
            RunningMeanStd(shape=(), device=self.device)
            if self.params.use_rms
            else None
        )

        self.collector = Collector(
            self.env,
            self.actor_module,
            total_frames=self.params.loops * self.params.steps_per_batch * self.n_envs,
            frames_per_batch=self.params.steps_per_batch * self.n_envs,
            device=self.device,
            env_device=self.device,
            policy_device=self.device,
            storing_device=self.device,
            no_cuda_sync=True,
        )

        if self.params.rnn_mode:
            if self.params.tbptt_steps is None:
                raise ValueError("tbptt_steps must be set in rnn_mode")
            if self.params.tbptt_steps > self.params.steps_per_batch:
                raise ValueError("tbptt_steps must be <= steps_per_batch")

    def train(self) -> None:
        for batch_idx, batch in enumerate(self.collector):
            logs = defaultdict(list)
            batch = batch.to(self.device)
            self._compute_gae(batch)

            for epoch in range(self.params.epochs_per_batch):
                if self.params.rnn_mode:
                    sub_batch = sample_seq_batch(
                        batch,
                        batch_size=self.params.sub_batch_size,
                        seq_len=self.params.tbptt_steps,
                    )
                    sub_batch_inf = self._inference_seq_batch(sub_batch)
                else:
                    sub_batch = sample_batch(
                        batch, batch_size=self.params.sub_batch_size
                    )
                    sub_batch_inf = self._inference_batch(sub_batch)

                mean_old = sub_batch.get(self.params.mean_key).detach()
                std_old = sub_batch.get(self.params.std_key).detach()
                mean_new = sub_batch_inf.get(self.params.mean_key)
                std_new = sub_batch_inf.get(self.params.std_key)

                dist_old = Normal(mean_old, std_old)
                dist_new = Normal(mean_new, std_new)
                kl_approx = kl_divergence(dist_old, dist_new).sum(-1).mean()

                if epoch > 0 and kl_approx.item() > 0.0:
                    if kl_approx.item() > self.params.desired_kl * 2.0:
                        self.actor_lr = max(
                            self.actor_lr / 1.5, self.params.actor_lr_min
                        )
                    elif kl_approx.item() < self.params.desired_kl / 2.0:
                        self.actor_lr = min(
                            self.actor_lr * 1.5, self.params.actor_lr_max
                        )

                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.actor_lr

                    if kl_approx.item() > self.params.upper_kl:
                        break

                action = sub_batch.get(self.params.action_key).detach()
                return_ = sub_batch.get(self.params.return_key).detach()
                value_old = sub_batch.get(self.params.value_key).detach()
                advantage = sub_batch.get(self.params.advantage_key).detach()
                value = sub_batch_inf.get(self.params.value_key)

                log_prob_old = dist_old.log_prob(action).sum(-1).unsqueeze(-1)
                log_prob_new = dist_new.log_prob(action).sum(-1).unsqueeze(-1)
                ratio = torch.exp(log_prob_new - log_prob_old)
                surrogate = ratio * advantage
                surrogate_clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.params.clip_epsilon,
                        1.0 + self.params.clip_epsilon,
                    )
                    * advantage
                )
                actor_loss = -torch.min(surrogate, surrogate_clipped).mean()
                entropy_loss = (
                    -dist_new.entropy().sum(-1).mean() * self.params.entropy_loss_coeff
                )

                value_clipped = value_old + torch.clamp(
                    value - value_old,
                    -self.params.value_clip_epsilon,
                    self.params.value_clip_epsilon,
                )
                if self.params.value_loss_type == "smooth_l1":
                    value_loss_unclipped = torch.nn.functional.smooth_l1_loss(
                        value, return_, reduction="none"
                    )
                    value_loss_clipped = torch.nn.functional.smooth_l1_loss(
                        value_clipped, return_, reduction="none"
                    )
                elif self.params.value_loss_type == "l1":
                    value_loss_unclipped = torch.nn.functional.l1_loss(
                        value, return_, reduction="none"
                    )
                    value_loss_clipped = torch.nn.functional.l1_loss(
                        value_clipped, return_, reduction="none"
                    )
                elif self.params.value_loss_type == "l2":
                    value_loss_unclipped = torch.nn.functional.mse_loss(
                        value, return_, reduction="none"
                    )
                    value_loss_clipped = torch.nn.functional.mse_loss(
                        value_clipped, return_, reduction="none"
                    )
                else:
                    raise ValueError(
                        f"Invalid value_loss_type: {self.params.value_loss_type}"
                    )

                critic_loss = (
                    torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    * self.params.critic_loss_coeff
                )

                total_loss = actor_loss + entropy_loss + critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_network.parameters(), self.params.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(), self.params.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                logs["actor_loss"].append(actor_loss.item())
                logs["critic_loss"].append(critic_loss.item())
                logs["entropy_loss"].append(entropy_loss.item())
                logs["kl_approx"].append(kl_approx.item())
                logs["actor_lr"].append(self.actor_lr)
                logs["batch_trained"].append(1)

            if self.logger is not None:
                reset_mean = (
                    batch["next"]
                    .get(self.params.reset_key)
                    .to(torch.float32)
                    .flatten()
                    .mean()
                    .item()
                )
                terminate_mean = (
                    batch["next"]
                    .get(self.params.terminate_key)
                    .to(torch.float32)
                    .flatten()
                    .mean()
                    .item()
                )

                def _safe_mean(vals: list[float]) -> float:
                    return sum(vals) / len(vals) if vals else 0.0

                log_dict = {
                    f"{self.params.task_name}/actor_loss": _safe_mean(
                        logs["actor_loss"]
                    ),
                    f"{self.params.task_name}/critic_loss": _safe_mean(
                        logs["critic_loss"]
                    ),
                    f"{self.params.task_name}/entropy_loss": _safe_mean(
                        logs["entropy_loss"]
                    ),
                    f"{self.params.task_name}/kl_approx": _safe_mean(logs["kl_approx"]),
                    f"{self.params.task_name}/batch_num": sum(logs["batch_trained"]),
                    f"{self.params.task_name}/actor_lr": _safe_mean(logs["actor_lr"]),
                    f"{self.params.task_name}/reward": batch["next"]
                    .get(self.params.reward_key)
                    .mean()
                    .item(),
                    f"{self.params.task_name}/episode_length": 1.0
                    / (reset_mean + 1e-10),
                    f"{self.params.task_name}/terminate_prob": terminate_mean,
                }

                info: TensorDict = batch.get(self.params.info_keys, default=None)
                if info is not None:
                    for nkey in info.keys(include_nested=True, leaves_only=True):
                        if isinstance(nkey, tuple):
                            key = "/".join(nkey)
                        else:
                            key = str(nkey)
                        value = info.get(nkey).mean().item()
                        log_dict[f"{self.params.task_name}/{key}"] = value

                for key, value in log_dict.items():
                    self.logger.log_scalar(key, value, step=batch_idx)

            if (batch_idx + 1) % self.params.checkpoint_interval == 0:
                exp_name = getattr(self.logger, "exp_name", "default")
                path = f"{self.params.checkpoint_dir}/{exp_name}"
                os.makedirs(path, exist_ok=True)
                torch.save(
                    {
                        "batch_idx": batch_idx,
                        "task_name": self.params.task_name,
                        "exp_name": exp_name,
                        "actor_network": self.actor_network.state_dict(),
                        "critic_network": self.critic_network.state_dict(),
                        "actor_optimizer": self.actor_optimizer.state_dict(),
                        "critic_optimizer": self.critic_optimizer.state_dict(),
                    },
                    f"{path}/{exp_name}_{batch_idx + 1}.pth",
                )

    @staticmethod
    def _dedupe_keys(keys: list[NestedKey]) -> list[NestedKey]:
        deduped: list[NestedKey] = []
        for key in keys:
            if key not in deduped:
                deduped.append(key)
        return deduped

    def _inference_batch(
        self, batch: TensorDict, value_only: bool = False
    ) -> TensorDict:
        if value_only:
            in_keys = list(self.critic_module.in_keys)
        else:
            in_keys = self._dedupe_keys(
                list(self.actor_module.in_keys) + list(self.critic_module.in_keys)
            )

        batch_inf = batch.select(*in_keys).clone()
        batch_inf = self.critic_module(batch_inf)
        if not value_only:
            batch_inf = self.actor_module(batch_inf)
        return batch_inf

    def _inference_seq_batch(
        self, batch: TensorDict, value_only: bool = False
    ) -> TensorDict:
        _b, t = batch.shape[:2]
        batch_inf_steps = []

        if value_only:
            in_keys = list(self.critic_module.in_keys)
        else:
            in_keys = self._dedupe_keys(
                list(self.actor_module.in_keys) + list(self.critic_module.in_keys)
            )

        for i in range(t):
            step_batch_inf = batch[:, i].select(*in_keys).clone()
            if i > 0:
                step_next = batch_inf_steps[i - 1].get("next", default=None)
                if step_next is not None:
                    for key in step_next.keys(include_nested=True, leaves_only=True):
                        step_batch_inf.set(key, step_next.get(key))

            step_batch_inf = self.critic_module(step_batch_inf)
            if not value_only:
                step_batch_inf = self.actor_module(step_batch_inf)
            batch_inf_steps.append(step_batch_inf)

        batch_inf = tensordict.stack(batch_inf_steps, dim=1)
        if batch_inf.shape[:2] != (_b, t):
            raise RuntimeError("Unexpected sequence inference output shape")
        return batch_inf

    @torch.no_grad()
    def _compute_gae(self, batch: TensorDict) -> TensorDict:
        if (
            batch.shape[0] != self.env.batch_size[0]
            or batch.shape[1] != self.params.steps_per_batch
        ):
            raise ValueError("Batch shape must match (n_envs, steps_per_batch)")

        batch_next = batch.get("next")
        if batch_next is None:
            raise ValueError("batch must contain 'next' key for GAE computation")

        rewards = batch_next.get(self.params.reward_key)
        resets = batch_next.get(self.params.reset_key)
        terminates = batch_next.get(self.params.terminate_key)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        if self.params.rnn_mode:
            values_raw = self._inference_seq_batch(batch, value_only=True).get(
                self.params.value_key
            )
            next_values_raw = self._inference_seq_batch(
                batch_next, value_only=True
            ).get(self.params.value_key)
        else:
            values_raw = self._inference_batch(batch, value_only=True).get(
                self.params.value_key
            )
            next_values_raw = self._inference_batch(batch_next, value_only=True).get(
                self.params.value_key
            )

        if self.rms is not None:
            values = self.rms.denormalize(values_raw)
            next_values = self.rms.denormalize(next_values_raw)
        else:
            values = values_raw
            next_values = next_values_raw

        reset_mask = resets.logical_not().to(torch.float32)
        terminate_mask = terminates.logical_not().to(torch.float32)
        delta = rewards + self.params.gamma * next_values * terminate_mask - values

        for t in reversed(range(rewards.shape[1])):
            if t == rewards.shape[1] - 1:
                advantages[:, t] = delta[:, t]
            else:
                advantages[:, t] = (
                    delta[:, t]
                    + self.params.gamma
                    * self.params.lam
                    * advantages[:, t + 1]
                    * reset_mask[:, t]
                )
            returns[:, t] = advantages[:, t] + values[:, t]

        if self.rms is not None:
            self.rms.update(returns)
            returns = self.rms.normalize(returns)

        if self.params.normalize_advantages:
            eps = 1e-10
            if self.params.normalize_advantages_type == "standard":
                advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
            elif self.params.normalize_advantages_type == "std_only":
                advantages = advantages / (advantages.std() + eps)
            elif self.params.normalize_advantages_type == "rms":
                advantages = advantages / (advantages.square().mean().sqrt() + eps)
            else:
                raise ValueError(
                    f"Invalid normalize_advantages_type: {self.params.normalize_advantages_type}"
                )

        batch.set(self.params.value_key, values_raw)
        batch.set(self.params.advantage_key, advantages)
        batch.set(self.params.return_key, returns)
        return batch

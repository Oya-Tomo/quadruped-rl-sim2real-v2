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
class TRPOParams:
	task_name: str
	loops: int
	steps_per_batch: int
	sub_batch_size: int
	critic_iters: int
	checkpoint_interval: int
	checkpoint_dir: str
	gamma: float
	lam: float
	max_kl: float
	cg_iters: int
	cg_damping: float
	backtrack_steps: int
	backtrack_coeff: float
	improve_ratio_threshold: float
	entropy_loss_coeff: float
	critic_loss_coeff: float
	value_loss_type: Literal["smooth_l1", "l1", "l2"]
	use_rms: bool
	max_grad_norm: float
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


class TRPOTrainer:
	def __init__(
		self,
		params: TRPOParams,
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

			actor_loss, entropy_loss, kl_value, step_norm, accepted = (
				self._update_actor(batch)
			)
			logs["actor_loss"].append(actor_loss)
			logs["entropy_loss"].append(entropy_loss)
			logs["kl_approx"].append(kl_value)
			logs["step_norm"].append(step_norm)
			logs["line_search_accepted"].append(float(accepted))

			for _ in range(self.params.critic_iters):
				if self.params.rnn_mode:
					sub_batch = sample_seq_batch(
						batch,
						batch_size=self.params.sub_batch_size,
						seq_len=self.params.tbptt_steps,
					)
					sub_batch_inf = self._inference_seq_batch(sub_batch, value_only=True)
				else:
					sub_batch = sample_batch(batch, batch_size=self.params.sub_batch_size)
					sub_batch_inf = self._inference_batch(sub_batch, value_only=True)

				return_ = sub_batch.get(self.params.return_key).detach()
				value = sub_batch_inf.get(self.params.value_key)

				if self.params.value_loss_type == "smooth_l1":
					value_loss = torch.nn.functional.smooth_l1_loss(value, return_)
				elif self.params.value_loss_type == "l1":
					value_loss = torch.nn.functional.l1_loss(value, return_)
				elif self.params.value_loss_type == "l2":
					value_loss = torch.nn.functional.mse_loss(value, return_)
				else:
					raise ValueError(
						f"Invalid value_loss_type: {self.params.value_loss_type}"
					)

				critic_loss = value_loss * self.params.critic_loss_coeff

				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.critic_network.parameters(), self.params.max_grad_norm
				)
				self.critic_optimizer.step()

				logs["critic_loss"].append(critic_loss.item())
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
					f"{self.params.task_name}/step_norm": _safe_mean(logs["step_norm"]),
					f"{self.params.task_name}/line_search_accepted": _safe_mean(
						logs["line_search_accepted"]
					),
					f"{self.params.task_name}/batch_num": sum(logs["batch_trained"]),
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

	def _get_actor_dist(
		self, batch: TensorDict
	) -> tuple[Normal, torch.Tensor, torch.Tensor]:
		if self.params.rnn_mode:
			batch_inf = self._inference_seq_batch(batch)
		else:
			batch_inf = self._inference_batch(batch)

		mean = batch_inf.get(self.params.mean_key)
		std = batch_inf.get(self.params.std_key)
		return Normal(mean, std), mean, std

	@staticmethod
	def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
		return torch.cat([t.reshape(-1) for t in tensors])

	def _flat_actor_params(self) -> torch.Tensor:
		params = [p.data for p in self.actor_network.parameters()]
		return self._flatten_tensors(params)

	def _set_flat_actor_params(self, flat_params: torch.Tensor):
		offset = 0
		for p in self.actor_network.parameters():
			numel = p.numel()
			p.data.copy_(flat_params[offset : offset + numel].view_as(p))
			offset += numel

	def _flat_actor_grad(
		self,
		loss: torch.Tensor,
		retain_graph: bool = False,
		create_graph: bool = False,
	) -> torch.Tensor:
		params = list(self.actor_network.parameters())
		grads = torch.autograd.grad(
			loss,
			params,
			retain_graph=retain_graph,
			create_graph=create_graph,
			allow_unused=False,
		)
		return self._flatten_tensors(list(grads))

	def _fisher_vector_product(
		self,
		kl: torch.Tensor,
		vector: torch.Tensor,
	) -> torch.Tensor:
		grad_kl = self._flat_actor_grad(kl, retain_graph=True, create_graph=True)
		grad_kl_v = (grad_kl * vector).sum()
		flat_grad2 = self._flat_actor_grad(grad_kl_v, retain_graph=True)
		return flat_grad2 + self.params.cg_damping * vector

	def _conjugate_gradient(
		self,
		fvp_fn,
		b: torch.Tensor,
		iters: int,
		tol: float = 1e-10,
	) -> torch.Tensor:
		x = torch.zeros_like(b)
		r = b.clone()
		p = r.clone()
		r_dot_r = torch.dot(r, r)

		for _ in range(iters):
			Ap = fvp_fn(p)
			denom = torch.dot(p, Ap) + 1e-8
			alpha = r_dot_r / denom
			x = x + alpha * p
			r = r - alpha * Ap
			new_r_dot_r = torch.dot(r, r)
			if new_r_dot_r < tol:
				break
			beta = new_r_dot_r / (r_dot_r + 1e-8)
			p = r + beta * p
			r_dot_r = new_r_dot_r
		return x

	def _surrogate_and_kl(
		self,
		batch: TensorDict,
		advantage: torch.Tensor,
		old_mean: torch.Tensor,
		old_std: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		dist_new, _, _ = self._get_actor_dist(batch)
		dist_old = Normal(old_mean, old_std)

		action = batch.get(self.params.action_key)
		log_prob_old = dist_old.log_prob(action).sum(dim=-1, keepdim=True)
		log_prob_new = dist_new.log_prob(action).sum(dim=-1, keepdim=True)
		ratio = torch.exp(log_prob_new - log_prob_old)

		entropy = dist_new.entropy().sum(dim=-1, keepdim=True)
		surrogate = (
			ratio * advantage + self.params.entropy_loss_coeff * entropy
		).mean()
		kl = kl_divergence(dist_old, dist_new).sum(dim=-1, keepdim=True).mean()
		return surrogate, kl, entropy.mean()

	def _update_actor(self, batch: TensorDict) -> tuple[float, float, float, float, bool]:
		advantage = batch.get(self.params.advantage_key).detach()
		old_mean = batch.get(self.params.mean_key).detach()
		old_std = batch.get(self.params.std_key).detach()

		surrogate_old, kl_old, entropy_old = self._surrogate_and_kl(
			batch, advantage, old_mean, old_std
		)
		del kl_old

		actor_loss = -surrogate_old
		grad = self._flat_actor_grad(actor_loss, retain_graph=True).detach()

		fvp_fn = lambda v: self._fisher_vector_product(
			self._surrogate_and_kl(batch, advantage, old_mean, old_std)[1], v
		).detach()
		step_dir = self._conjugate_gradient(fvp_fn, -grad, iters=self.params.cg_iters)

		fvp_step = fvp_fn(step_dir)
		shs = 0.5 * torch.dot(step_dir, fvp_step)
		step_scale = torch.sqrt(
			torch.clamp(self.params.max_kl / (shs + 1e-8), min=0.0)
		)
		full_step = step_dir * step_scale
		expected_improve = torch.dot(-grad, full_step)

		old_params = self._flat_actor_params().detach().clone()
		old_surrogate = surrogate_old.detach()

		accepted = False
		final_kl = 0.0
		step_norm = 0.0
		final_entropy = entropy_old.detach().item()

		for i in range(self.params.backtrack_steps):
			frac = self.params.backtrack_coeff**i
			new_params = old_params + frac * full_step
			self._set_flat_actor_params(new_params)

			with torch.no_grad():
				new_surrogate, new_kl, new_entropy = self._surrogate_and_kl(
					batch, advantage, old_mean, old_std
				)

			actual_improve = new_surrogate - old_surrogate
			expected_improve_frac = expected_improve * frac
			improve_ratio = actual_improve / (expected_improve_frac + 1e-8)

			if (
				new_kl.item() <= self.params.max_kl
				and actual_improve.item() > 0.0
				and improve_ratio.item() >= self.params.improve_ratio_threshold
			):
				accepted = True
				final_kl = new_kl.item()
				step_norm = (frac * full_step).norm().item()
				final_entropy = new_entropy.item()
				break

		if not accepted:
			self._set_flat_actor_params(old_params)
			with torch.no_grad():
				_, fallback_kl, fallback_entropy = self._surrogate_and_kl(
					batch, advantage, old_mean, old_std
				)
			final_kl = fallback_kl.item()
			final_entropy = fallback_entropy.item()

		with torch.no_grad():
			final_surrogate, _, _ = self._surrogate_and_kl(
				batch, advantage, old_mean, old_std
			)
			final_actor_loss = (-final_surrogate).item()

		return final_actor_loss, final_entropy, final_kl, step_norm, accepted

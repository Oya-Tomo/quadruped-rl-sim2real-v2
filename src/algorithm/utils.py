from __future__ import annotations

import tensordict
import torch
from tensordict import TensorDict


def sample_batch(batch: TensorDict, batch_size: int) -> TensorDict:
    if batch.ndim < 2:
        raise ValueError("batch must have at least (env, time) dimensions")

    flat_batch = batch.reshape(-1)
    total = flat_batch.batch_size[0]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = torch.randint(total, (batch_size,), device=flat_batch.device)
    return flat_batch[indices]


def sample_seq_batch(batch: TensorDict, batch_size: int, seq_len: int) -> TensorDict:
    if batch.ndim < 2:
        raise ValueError("batch must have at least (env, time) dimensions")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    n_envs, time_steps = batch.batch_size[:2]
    if seq_len > time_steps:
        raise ValueError("seq_len must be less than or equal to time dimension")

    env_indices = torch.randint(n_envs, (batch_size,), device=batch.device)
    start_max = time_steps - seq_len + 1
    start_indices = torch.randint(start_max, (batch_size,), device=batch.device)

    seq_batches = []
    for env_idx, start_idx in zip(env_indices.tolist(), start_indices.tolist()):
        seq_batches.append(batch[env_idx, start_idx : start_idx + seq_len])

    return tensordict.stack(seq_batches, dim=0)

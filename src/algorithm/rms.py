from __future__ import annotations

import torch


class RunningMeanStd:
    def __init__(
        self,
        shape: tuple[int, ...] | int | None = (),
        device: torch.device | str | None = None,
        epsilon: float = 1e-4,
    ) -> None:
        if shape is None:
            shape = ()
        if isinstance(shape, int):
            shape = (shape,)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.to(dtype=torch.float32)
        if x.numel() == 0:
            return

        reduce_dims = tuple(range(max(0, x.ndim - self.mean.ndim)))
        batch_mean = x.mean(dim=reduce_dims)
        batch_var = x.var(dim=reduce_dims, unbiased=False)
        if reduce_dims:
            sample_count = 1
            for dim in reduce_dims:
                sample_count *= x.shape[dim]
        else:
            sample_count = 1
        batch_count = torch.tensor(
            sample_count,
            dtype=torch.float32,
            device=x.device,
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + eps)

    def denormalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x * torch.sqrt(self.var + eps) + self.mean

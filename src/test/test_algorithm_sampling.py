from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict

from src.algorithm.utils import sample_batch, sample_seq_batch


class AlgorithmSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_envs = 4
        self.time_steps = 7
        obs = torch.arange(
            self.n_envs * self.time_steps * 3, dtype=torch.float32
        ).reshape(self.n_envs, self.time_steps, 3)
        rew = torch.arange(self.n_envs * self.time_steps, dtype=torch.float32).reshape(
            self.n_envs, self.time_steps, 1
        )
        self.batch = TensorDict(
            {
                "obs": obs,
                ("next", "reward"): rew,
            },
            batch_size=(self.n_envs, self.time_steps),
        )

    def test_sample_batch_shape_and_keys(self) -> None:
        sampled = sample_batch(self.batch, batch_size=5)
        self.assertEqual(tuple(sampled.batch_size), (5,))
        self.assertIn("obs", sampled.keys())
        self.assertIsNotNone(sampled.get(("next", "reward"), None))
        self.assertEqual(tuple(sampled["obs"].shape), (5, 3))

    def test_sample_seq_batch_shape_and_keys(self) -> None:
        seq_len = 3
        sampled = sample_seq_batch(self.batch, batch_size=6, seq_len=seq_len)
        self.assertEqual(tuple(sampled.batch_size), (6, seq_len))
        self.assertIn("obs", sampled.keys())
        self.assertIsNotNone(sampled.get(("next", "reward"), None))
        self.assertEqual(tuple(sampled["obs"].shape), (6, seq_len, 3))

    def test_sample_seq_batch_invalid_seq_len(self) -> None:
        with self.assertRaises(ValueError):
            sample_seq_batch(self.batch, batch_size=2, seq_len=self.time_steps + 1)

    def test_sample_batch_invalid_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            sample_batch(self.batch, batch_size=0)


if __name__ == "__main__":
    unittest.main()

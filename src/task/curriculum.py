import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch

from .field import (
    HeightMapField,
    random_tile_terrain_height_map,
    stair_terrain_height_map,
)


def _height_map_field_from_tensor(
    height_map: torch.Tensor,
    cell_size: float,
) -> HeightMapField:
    field = HeightMapField(
        cell_num_x=height_map.shape[1] - 1,
        cell_num_y=height_map.shape[0] - 1,
        cell_size=cell_size,
        device=height_map.device,
    )
    field.height_map = height_map.clone()
    return field


@dataclass(frozen=True)
class TerrainCurriculumConfig:
    num_rows: int
    num_cols: int
    cell_size: float
    stair_step_tread: float
    stair_step_num: int = 10
    tile_num: int = 32
    tile_size_min: int = 2
    tile_size_max: int = 5
    tile_min_height: float = 0.0
    tile_max_height: float | None = None
    mix_ratio: float = 0.5
    seed: int | None = None
    stair_direction: str = "x"


@dataclass(frozen=True)
class MixedTerrainLevel:
    level_index: int
    step_height: float
    stair_field: HeightMapField
    random_tile_field: HeightMapField
    mix_ratio: float = 0.5

    def sample_field(self, episode_index: int) -> HeightMapField:
        if self.mix_ratio <= 0.0:
            return self.random_tile_field
        if self.mix_ratio >= 1.0:
            return self.stair_field

        rng_seed = (self.level_index + 1) * 1_000_003 + episode_index
        rng = random.Random(rng_seed)
        return (
            self.stair_field
            if rng.random() < self.mix_ratio
            else self.random_tile_field
        )


def build_mixed_terrain_level(
    level_index: int,
    step_height: float,
    config: TerrainCurriculumConfig,
) -> MixedTerrainLevel:
    tile_max_height = (
        step_height if config.tile_max_height is None else config.tile_max_height
    )
    if tile_max_height < config.tile_min_height:
        raise ValueError("tile_max_height must be >= tile_min_height")

    seed_base = None if config.seed is None else config.seed + level_index * 10_000

    stair_height_map = stair_terrain_height_map(
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        cell_size=config.cell_size,
        step_height=step_height,
        step_tread=config.stair_step_tread,
        step_num=config.stair_step_num,
        direction=config.stair_direction,
        seed=seed_base,
    )
    random_tile_height_map = random_tile_terrain_height_map(
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        tile_num=config.tile_num,
        tile_size_min=config.tile_size_min,
        tile_size_max=config.tile_size_max,
        min_height=config.tile_min_height,
        max_height=tile_max_height,
        seed=None if seed_base is None else seed_base + 1,
    )

    return MixedTerrainLevel(
        level_index=level_index,
        step_height=step_height,
        stair_field=_height_map_field_from_tensor(stair_height_map, config.cell_size),
        random_tile_field=_height_map_field_from_tensor(
            random_tile_height_map, config.cell_size
        ),
        mix_ratio=config.mix_ratio,
    )


def build_step_height_curriculum(
    step_heights: Sequence[float],
    config: TerrainCurriculumConfig,
) -> list[MixedTerrainLevel]:
    return [
        build_mixed_terrain_level(
            level_index=level_index, step_height=step_height, config=config
        )
        for level_index, step_height in enumerate(step_heights)
    ]


@dataclass(frozen=True)
class TrackingScoreConfig:
    min_command_speed: float = 0.5
    success_error_threshold: float = 0.25
    command_speed_dimensions: tuple[int, int] = (0, 1)
    tracked_dimensions: tuple[int, ...] = (0, 1, 2)
    min_valid_steps: int = 1


class EpisodeTrackingAccumulator:
    def __init__(self, config: TrackingScoreConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.valid_steps = 0
        self.valid_error_sum = 0.0

    def update(self, command: torch.Tensor, measured: torch.Tensor):
        if command.shape != measured.shape:
            raise ValueError("command and measured must have the same shape")

        command = command.detach()
        measured = measured.detach()

        command_speed = torch.linalg.norm(
            command[..., self.config.command_speed_dimensions], dim=-1
        )
        valid_mask = command_speed >= self.config.min_command_speed
        tracking_error = torch.linalg.norm(
            command[..., self.config.tracked_dimensions]
            - measured[..., self.config.tracked_dimensions],
            dim=-1,
        )

        self.total_steps += tracking_error.numel()
        self.valid_steps += int(valid_mask.sum().item())
        self.valid_error_sum += float(tracking_error[valid_mask].sum().item())

    def compute(self) -> dict[str, float | int | bool]:
        mean_valid_error = self.valid_error_sum / max(self.valid_steps, 1)
        valid_step_ratio = self.valid_steps / max(self.total_steps, 1)
        episode_success = (
            self.valid_steps >= self.config.min_valid_steps
            and mean_valid_error <= self.config.success_error_threshold
        )
        return {
            "mean_tracking_error": mean_valid_error,
            "valid_step_ratio": valid_step_ratio,
            "valid_step_count": self.valid_steps,
            "total_step_count": self.total_steps,
            "episode_success": episode_success,
        }


@dataclass(frozen=True)
class CurriculumControllerConfig:
    promotion_success_rate: float = 0.7
    demotion_success_rate: float = 0.3
    window_size: int = 100


class CurriculumController:
    def __init__(
        self,
        levels: Sequence[MixedTerrainLevel],
        config: CurriculumControllerConfig,
        initial_level_index: int = 0,
    ):
        if not levels:
            raise ValueError("levels must not be empty")
        if config.window_size <= 0:
            raise ValueError("window_size must be positive")

        self.levels = list(levels)
        self.config = config
        self.current_level_index = max(
            0, min(initial_level_index, len(self.levels) - 1)
        )
        self.success_history = deque(maxlen=config.window_size)

    @property
    def current_level(self) -> MixedTerrainLevel:
        return self.levels[self.current_level_index]

    @property
    def success_rate(self) -> float:
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)

    def record_episode_success(self, success: bool) -> dict[str, float | int | bool]:
        self.success_history.append(bool(success))

        level_changed = False
        if len(self.success_history) >= self.success_history.maxlen:
            rate = self.success_rate
            if (
                rate >= self.config.promotion_success_rate
                and self.current_level_index < len(self.levels) - 1
            ):
                self.current_level_index += 1
                level_changed = True
            elif (
                rate <= self.config.demotion_success_rate
                and self.current_level_index > 0
            ):
                self.current_level_index -= 1
                level_changed = True

        if level_changed:
            self.success_history.clear()

        return {
            "current_level_index": self.current_level_index,
            "success_rate": self.success_rate,
            "level_changed": level_changed,
            "history_size": len(self.success_history),
        }

    def sample_field_for_episode(self, episode_index: int) -> HeightMapField:
        return self.current_level.sample_field(episode_index)

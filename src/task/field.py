import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Self

import genesis as gs
import torch
import torch.nn.functional as F


@contextmanager
def _seeded_rng(seed: int | None):
    if seed is None:
        yield
        return

    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


class Field(ABC):
    @abstractmethod
    def get_height(self, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_bounds(
        self,
    ) -> tuple[float, float, float, float]:  # min_x, max_x, min_y, max_y
        raise NotImplementedError()

    @abstractmethod
    def create_entities(
        self, offset_x: float = 0.0, offset_y: float = 0.0
    ) -> list[gs.morphs.Morph]:
        raise NotImplementedError()

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        raise NotImplementedError()


class PlaneField(Field):
    def __init__(self, height: float = 0.0):
        self.height = height

    @property
    def config(self) -> dict:
        return {
            "type": "plane",
            "height": self.height,
        }

    def get_height(self, positions: torch.Tensor) -> torch.Tensor:
        shape = positions.shape[:-1]
        return torch.full(
            shape, self.height, device=positions.device, dtype=positions.dtype
        )

    def get_bounds(
        self,
    ) -> tuple[float, float, float, float]:
        return [-100.0, 100.0, -100.0, 100.0]

    def create_entities(
        self, offset_x: float = 0.0, offset_y: float = 0.0
    ) -> list[gs.morphs.Morph]:
        return [
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                pos=(offset_x, offset_y, self.height),
                fixed=True,
            )
        ]

    def to(self, device: torch.device):
        return self


class HeightMapField(Field):
    def __init__(
        self,
        cell_num_x: int,
        cell_num_y: int,
        cell_size: float,
        device: torch.device = "cpu",
    ):
        self.cell_num_x = cell_num_x
        self.cell_num_y = cell_num_y
        self.cell_size = cell_size
        self.width = cell_num_x * cell_size
        self.height = cell_num_y * cell_size

        self.height_map = torch.zeros(
            (cell_num_y + 1, cell_num_x + 1), dtype=torch.float32, device=device
        )

    @property
    def config(self) -> dict:
        return {
            "type": "height_map",
            "cell_num_x": self.cell_num_x,
            "cell_num_y": self.cell_num_y,
            "cell_size": self.cell_size,
            "width": self.width,
            "height": self.height,
        }

    def replace_height_map(
        self,
        height_map: torch.Tensor,
        start_cell_x: int,
        start_cell_y: int,
    ):
        assert (
            start_cell_y + height_map.shape[0] <= self.height_map.shape[0]
        ), "Height map row size exceeds."
        assert (
            start_cell_x + height_map.shape[1] <= self.height_map.shape[1]
        ), "Height map column size exceeds."
        self.height_map[
            start_cell_y : start_cell_y + height_map.shape[0],
            start_cell_x : start_cell_x + height_map.shape[1],
        ] = height_map.to(self.height_map.device)

    def get_height(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get height at the given positions.

        Parameters
            - positions: [x, y] positions, accepts coordinate tensors of any shape.
        """
        shape = positions.shape[:-1]
        pos_grid = positions.clone().flatten(0, -2)
        pos_grid[..., 0] = pos_grid[..., 0] / self.width * 2.0 - 1.0
        pos_grid[..., 1] = pos_grid[..., 1] / self.height * 2.0 - 1.0

        height = F.grid_sample(
            self.height_map.unsqueeze(0).unsqueeze(0),
            pos_grid.unsqueeze(0).unsqueeze(0),
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        return height.flatten(0, -1).view(*shape)

    def get_bounds(
        self,
    ) -> tuple[float, float, float, float]:
        return [0.0, self.width, 0.0, self.height]

    def create_entities(self) -> list[gs.morphs.Morph]:
        return [
            gs.morphs.Terrain(
                pos=(0.0, 0.0, 0.0),
                horizontal_scale=self.cell_size,
                vertical_scale=1.0,
                height_field=self.height_map.T.cpu().numpy(),
            )
        ]

    def to(self, device: torch.device) -> Self:
        field = HeightMapField(
            cell_num_x=self.cell_num_x,
            cell_num_y=self.cell_num_y,
            cell_size=self.cell_size,
            device=device,
        )
        field.height_map = self.height_map.to(device)
        return field


class PlaneBasedHeightMapField(Field):
    def __init__(
        self,
        cell_num_x: int,
        cell_num_y: int,
        cell_size: float,
        plane_height: float = -0.005,
        device: torch.device = "cpu",
    ):
        self.cell_num_x = cell_num_x
        self.cell_num_y = cell_num_y
        self.cell_size = cell_size
        self.width = cell_num_x * cell_size
        self.height = cell_num_y * cell_size
        self.plane_height = plane_height

        self.height_map_context = HeightMapField(
            cell_num_x=cell_num_x,
            cell_num_y=cell_num_y,
            cell_size=cell_size,
            device=device,
        )
        self.height_maps = []

    @property
    def config(self) -> dict:
        return {
            "type": "plane_based_height_map",
            "cell_num_x": self.cell_num_x,
            "cell_num_y": self.cell_num_y,
            "cell_size": self.cell_size,
            "width": self.width,
            "height": self.height,
            "plane_height": self.plane_height,
        }

    def get_height(self, positions):
        return self.height_map_context.get_height(positions)

    def get_bounds(self):
        return self.height_map_context.get_bounds()

    def create_entities(
        self, offset_x: float = 0.0, offset_y: float = 0.0
    ) -> list[gs.morphs.Morph]:
        entities = [
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                pos=(offset_x, offset_y, self.plane_height),
                fixed=True,
            )
        ]
        for height_map_field, x_offset, y_offset in self.height_maps:
            entities.append(
                gs.morphs.Terrain(
                    pos=(offset_x + x_offset, offset_y + y_offset, 0.0),
                    horizontal_scale=height_map_field.cell_size,
                    vertical_scale=1.0,
                    height_field=height_map_field.height_map.T.cpu().numpy(),
                )
            )
        return entities

    def add_height_map_field(
        self,
        height_map_field: HeightMapField,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
    ):
        self.height_maps.append((height_map_field, x_offset, y_offset))
        self.height_map_context.replace_height_map(
            height_map_field.height_map,
            start_cell_x=int(x_offset / self.cell_size),
            start_cell_y=int(y_offset / self.cell_size),
        )

    def to(self, device: torch.device) -> Self:
        self.height_map_context = self.height_map_context.to(device)
        return self


def pyramid_terrain_height_map(
    num_rows: int,
    num_cols: int,
    cell_size: float,
    step_height: float,
    step_tread: float,
    step_num: int = 10,
) -> torch.Tensor:
    """
    Generate a pyramid terrain height map.
    """
    height_map = torch.zeros(
        (num_rows + 1, num_cols + 1), dtype=torch.float32, device="cpu"
    )

    for row in range(num_rows + 1):
        for col in range(num_cols + 1):
            cell_center_x = col * cell_size
            cell_center_y = row * cell_size
            height_map[row, col] = step_height * (
                min(
                    cell_center_x // step_tread,
                    (num_cols * cell_size - cell_center_x) // step_tread,
                    cell_center_y // step_tread,
                    (num_rows * cell_size - cell_center_y) // step_tread,
                    step_num,
                )
            )
    return height_map


def basin_terrain_height_map(
    num_rows: int,
    num_cols: int,
    cell_size: float,
    step_height: float,
    step_tread: float,
    step_num: int = 10,
) -> torch.Tensor:
    """
    Generate a basin terrain height map.
    """
    height_map = torch.zeros(
        (num_rows + 1, num_cols + 1), dtype=torch.float32, device="cpu"
    )
    peak_step = step_num // 2

    for row in range(num_rows + 1):
        for col in range(num_cols + 1):
            d_x = min(col, num_cols - col) * cell_size
            d_y = min(row, num_rows - row) * cell_size
            dist_from_edge = min(d_x, d_y)

            step_idx = int(dist_from_edge // step_tread)
            if step_idx <= peak_step:
                current_step = step_idx
            else:
                current_step = max(0, peak_step - (step_idx - peak_step))
            height_map[row, col] = current_step * step_height
    return height_map


def uniform_terrain_height_map(
    num_rows: int,
    num_cols: int,
    min_height: float,
    max_height: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate a uniform random terrain height map.
    """
    with _seeded_rng(seed):
        height_map = torch.randn((num_rows + 1, num_cols + 1), device="cpu")
        height_map = height_map * (max_height - min_height) + min_height
        height_map[0, :] = 0.0
        height_map[-1, :] = 0.0
        height_map[:, 0] = 0.0
        height_map[:, -1] = 0.0
        return height_map


def tile_terrain_height_map(
    num_rows: int,
    num_cols: int,
    tile_rows: int,
    tile_cols: int,
    min_height: float,
    max_height: float,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Tile the terrain height map.
    """
    with _seeded_rng(seed):
        height_map = torch.zeros((num_rows + 1, num_cols + 1), device="cpu")

        tiled_height_map = torch.randn((tile_rows, tile_cols), device="cpu")
        tiled_height_map = tiled_height_map * (max_height - min_height) + min_height

        for row in range(num_rows + 1):
            for col in range(num_cols + 1):
                if row % tile_rows == 0 or col % tile_cols == 0:
                    continue
                tiled_row = row // tile_rows
                tiled_col = col // tile_cols
                height_map[row, col] = tiled_height_map[tiled_row, tiled_col]

        return height_map


def random_tile_terrain_height_map(
    num_rows: int,
    num_cols: int,
    tile_num: int,
    tile_size_min: int,
    tile_size_max: int,
    min_height: float,
    max_height: float,
    seed: int | None = None,
) -> torch.Tensor:
    with _seeded_rng(seed):
        height_map = torch.zeros((num_rows + 1, num_cols + 1), device="cpu")
        for _ in range(tile_num):
            tile_size = random.randint(tile_size_min, tile_size_max)
            tile_start_row = random.randint(1, num_rows - tile_size - 1)
            tile_start_col = random.randint(1, num_cols - tile_size - 1)

            tile_height = random.random() * (max_height - min_height) + min_height
            height_map[
                tile_start_row : tile_start_row + tile_size,
                tile_start_col : tile_start_col + tile_size,
            ] = tile_height

        return height_map


def stair_terrain_height_map(
    num_rows: int,
    num_cols: int,
    cell_size: float,
    step_height: float,
    step_tread: float,
    step_num: int = 10,
    direction: str = "x",
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate a stair terrain height map that increases from one edge.
    """
    with _seeded_rng(seed):
        height_map = torch.zeros(
            (num_rows + 1, num_cols + 1), dtype=torch.float32, device="cpu"
        )

        for row in range(num_rows + 1):
            for col in range(num_cols + 1):
                if direction == "x":
                    distance = col * cell_size
                elif direction == "y":
                    distance = row * cell_size
                else:
                    raise ValueError(f"Unsupported stair direction: {direction}")

                height_map[row, col] = step_height * min(
                    distance // step_tread,
                    step_num,
                )

        return height_map

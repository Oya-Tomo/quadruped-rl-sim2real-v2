from __future__ import annotations

import math
import unittest

import genesis as gs
import torch
from torchrl.envs import TransformedEnv

from src.model.mlp import ActorNetwork as MLPActorNetwork
from src.model.mlp import ActorParams as MLPActorParams
from src.model.mlp import CriticNetwork as MLPCriticNetwork
from src.model.mlp import CriticParams as MLPCriticParams
from src.model.rnn import ActorNetwork as RNNActorNetwork
from src.model.rnn import ActorParams as RNNActorParams
from src.model.rnn import CriticNetwork as RNNCriticNetwork
from src.model.rnn import CriticParams as RNNCriticParams
from src.task.field import PlaneField
from src.task.walking_terrain import (
    CommandParams,
    ControlParams,
    RobotParams,
    SimulationParams,
    TerrainCurriculumParams,
    TestParams,
    WalkingTerrainEnv,
)


def _init_genesis_gpu() -> None:
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, seed=0)


def _make_env() -> WalkingTerrainEnv:
    _init_genesis_gpu()

    sim_params = SimulationParams(
        n_envs=1,
        frequency=50,
        substeps=4,
        max_episode_length=1000,
        shuffle_reset=False,
        begin_reset_episode_length=100,
    )
    robot_params = RobotParams(
        urdf="urdf/go2/urdf/go2.urdf",
        scale=1.0,
        init_pos=[0.0, 0.0, 0.5],
        init_quat=[1.0, 0.0, 0.0, 0.0],
        links_to_keep=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
    )
    control_params = ControlParams(
        dofs_names=[
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        dofs_origin=[
            0.0,
            0.8,
            -1.5,
            0.0,
            0.8,
            -1.5,
            0.0,
            1.0,
            -1.5,
            0.0,
            1.0,
            -1.5,
        ],
        dofs_kp=[40.0] * 12,
        dofs_kv=[1.0] * 12,
        ema_alpha=0.8,
    )
    command_params = CommandParams(
        x_vel_min=-1.5,
        x_vel_max=1.5,
        y_vel_min=-1.5,
        y_vel_max=1.5,
        yaw_vel_min=-2.0,
        yaw_vel_max=2.0,
        body_height_min=0.32,
        body_height_max=0.32,
        gait_period_min=0.4,
        gait_period_max=0.6,
        gait_phase_offset=[0.0, math.pi, math.pi, 0.0],
        resample_interval_avg=500,
    )
    terrain_curriculum_params = TerrainCurriculumParams(
        respawn_height=0.5,
        respawn_range=[(0.0, 0.0, 0.0, 0.0)],
        traversability_velocity_threshold=0.5,
        move_up_traversability=0.7,
        move_down_traversability=0.3,
        traversability_measure_steps_threshold=1,
        fall_down_threshold_radians=math.pi / 2,
    )

    return WalkingTerrainEnv(
        sim_params=sim_params,
        robot_params=robot_params,
        control_params=control_params,
        command_params=command_params,
        terrain_curriculum_params=terrain_curriculum_params,
        field=PlaneField(),
        test_params=TestParams.default_to_train(),
        device=torch.device("cuda:0"),
    )


def _state_keys() -> list[tuple[str, str]]:
    return [
        ("observation", "base_xyz_velocity"),
        ("observation", "base_rpy_velocity"),
        ("observation", "dofs_position"),
        ("observation", "dofs_velocity"),
        ("observation", "projected_gravity"),
        ("observation", "command"),
        ("observation", "action"),
        ("observation", "episode_progress"),
    ]


class ModelEnvPassTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_reset_tensordict_passes_mlp_modules(self) -> None:
        env = _make_env()
        try:
            td_reset = env.reset()
            td_for_mlp_actor = td_reset.clone()
            td_for_mlp_critic = td_reset.clone()

            state_keys = _state_keys()
            state_dim = 50
            action_dim = env.dofs_num

            actor = MLPActorNetwork(
                MLPActorParams(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=[128, 128],
                    state_keys=state_keys,
                    action_key="action",
                )
            ).to(env.device)
            critic = MLPCriticNetwork(
                MLPCriticParams(
                    state_dim=state_dim,
                    hidden_dims=[128, 128],
                    value_dim=1,
                    cost_dim=None,
                    state_keys=state_keys,
                    value_key="value",
                    value_cost_key=None,
                )
            ).to(env.device)

            td_actor = actor.create_module()(td_for_mlp_actor)
            td_critic = critic.create_module()(td_for_mlp_critic)

            self.assertTrue(td_actor.get("action", None) is not None)
            self.assertTrue(td_actor.get("mean", None) is not None)
            self.assertTrue(td_actor.get("std", None) is not None)
            self.assertTrue(td_critic.get("value", None) is not None)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            del env
            torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_reset_tensordict_passes_rnn_modules_with_primer(self) -> None:
        env = _make_env()
        transformed_env = None
        try:
            state_keys = _state_keys()
            state_dim = 50
            action_dim = env.dofs_num
            hidden_state_dim = 64
            hidden_state_key = "hidden_state"

            actor = RNNActorNetwork(
                RNNActorParams(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=[128, 128],
                    hidden_state_dim=hidden_state_dim,
                    state_keys=state_keys,
                    action_key="action",
                    hidden_state_key=hidden_state_key,
                    reset_hidden_key=None,
                )
            ).to(env.device)

            # Apply primer on the env side so reset output already has hidden state.
            transformed_env = TransformedEnv(env, actor.make_tensordict_primer())
            td_reset = transformed_env.reset()
            td_for_rnn_actor = td_reset.clone()
            td_for_rnn_critic = td_reset.clone()

            critic = RNNCriticNetwork(
                RNNCriticParams(
                    state_dim=state_dim,
                    hidden_dims=[128, 128],
                    hidden_state_dim=hidden_state_dim,
                    value_dim=1,
                    cost_dim=None,
                    state_keys=state_keys,
                    value_key="value",
                    value_cost_key=None,
                    hidden_state_key=hidden_state_key,
                    reset_hidden_key=None,
                )
            ).to(env.device)

            td_actor = actor.create_module()(td_for_rnn_actor)
            td_critic = critic.create_module()(td_for_rnn_critic)

            self.assertTrue(td_actor.get("action", None) is not None)
            self.assertTrue(td_actor.get("mean", None) is not None)
            self.assertTrue(td_actor.get("std", None) is not None)
            self.assertTrue(td_actor.get(("next", hidden_state_key), None) is not None)
            self.assertTrue(td_critic.get("value", None) is not None)
            self.assertTrue(td_critic.get(("next", hidden_state_key), None) is not None)
        finally:
            if transformed_env is not None:
                close = getattr(transformed_env, "close", None)
                if callable(close):
                    close()
            close = getattr(env, "close", None)
            if callable(close):
                close()
            del env
            torch.cuda.empty_cache()

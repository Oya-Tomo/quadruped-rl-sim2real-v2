from __future__ import annotations

import math
import unittest

import genesis as gs
import torch
from tensordict import TensorDict

from src.task.field import PlaneField
from src.task.walking_terrain import (
    CommandParams,
    ControlParams,
    RewardParams,
    RobotParams,
    SimulationParams,
    TerrainCurriculumParams,
    TestParams,
    WalkingTerrainEnv,
)


def _init_genesis_gpu() -> None:
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, seed=0)


def _make_env(use_all_reward_params: bool = False) -> WalkingTerrainEnv:
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
    test_params = TestParams(
        rendered_envs_idx=[0],
        camera_pos=(2.0, 2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.4),
        camera_fov=60.0,
        show_viewer=True,
        enable_shortcut=False,
    )

    reward_params = None
    if use_all_reward_params:
        reward_keys = [
            "xy_velocity_tracking",
            "yaw_velocity_tracking",
            "command_tracking",
            "curriculum_level",
        ]
        penalty_keys = [
            "body_height",
            "body_stability",
            "body_flip",
            "shoulder_height",
            "z_velocity",
            "roll_pitch_velocity",
            "dofs_velocity",
            "dofs_acceleration",
            "dofs_power",
            "feet_velocity",
            "feet_acceleration",
            "feet_slip",
            "feet_clearance",
            "feet_contact_release",
            "feet_soft_contact",
            "feet_contact_bias",
            "gait_pattern",
            "raibert_heuristic",
            "jumping_motion",
            "action_rate",
            "action_smoothness",
            "collision_dofs_limit",
        ]
        reward_params = RewardParams(
            reward_coef={key: 1.0 for key in reward_keys},
            penalty_coef={key: 1.0 for key in penalty_keys},
            shoulder_links=["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"],
            foot_links=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            terminate_components=[
                "fall_down",
                "unsafe_xyz_velocity",
                "unsafe_rpy_velocity",
            ],
            truncate_components=["timeout", "outside_field"],
        )

    return WalkingTerrainEnv(
        sim_params=sim_params,
        robot_params=robot_params,
        control_params=control_params,
        command_params=command_params,
        terrain_curriculum_params=terrain_curriculum_params,
        field=PlaneField(),
        test_params=test_params,
        device=torch.device("cuda:0"),
        reward_params=reward_params,
    )


class WalkingTerrainTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_zero_action_keeps_go2_upright(self) -> None:
        env = _make_env()
        try:
            env.reset()
            test_duration_seconds = 2.0
            n_steps = int(round(test_duration_seconds / env.dt))
            action = torch.zeros((1, env.dofs_num), device=env.device)

            for _ in range(n_steps):
                tensordict = TensorDict(
                    {"action": action},
                    batch_size=(1,),
                    device=env.device,
                )
                tensordict = env.step(tensordict)["next"]
                self.assertFalse(tensordict["terminated"].item())
                self.assertFalse(tensordict["truncated"].item())

            roll = env.state.base_rotation[0, 0].abs().item()
            pitch = env.state.base_rotation[0, 1].abs().item()
            self.assertLess(max(roll, pitch), 0.5)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            del env
            torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_random_rollout_go2(self) -> None:
        env = _make_env()
        try:
            env.reset()
            test_duration_seconds = 2.0
            n_steps = int(round(test_duration_seconds / env.dt))

            data = env.rollout(
                max_steps=n_steps,
                break_when_all_done=False,
                break_when_any_done=False,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            del env
            torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_all_reward_components_with_reward_params(self) -> None:
        env = _make_env(use_all_reward_params=True)
        try:
            td = env.reset()
            test_duration_seconds = 5.0
            n_steps = int(round(test_duration_seconds / env.dt))
            action = torch.zeros((1, env.dofs_num), device=env.device)
            for _ in range(n_steps):
                td = env.step(
                    TensorDict({"action": action}, batch_size=(1,), device=env.device)
                )["next"]

            info = td["info"]
            self.assertEqual(
                set(info.keys()), {"reward_components", "penalty_components"}
            )

            reward_components = info["reward_components"]
            expected_reward_keys = set(env.reward_params.reward_coef.keys()) | {
                "total"
            }
            self.assertEqual(set(reward_components.keys()), expected_reward_keys)
            for key in expected_reward_keys:
                value = reward_components[key]
                self.assertEqual(
                    tuple(value.shape), (1, 1), msg=f"shape mismatch: reward/{key}"
                )
                self.assertTrue(
                    torch.isfinite(value).all().item(),
                    msg=f"non-finite: reward/{key}",
                )

            penalty_components = info["penalty_components"]
            expected_penalty_keys = set(env.reward_params.penalty_coef.keys()) | {
                "total"
            }
            self.assertEqual(set(penalty_components.keys()), expected_penalty_keys)
            for key in expected_penalty_keys:
                value = penalty_components[key]
                self.assertEqual(
                    tuple(value.shape), (1, 1), msg=f"shape mismatch: penalty/{key}"
                )
                self.assertTrue(
                    torch.isfinite(value).all().item(),
                    msg=f"non-finite: penalty/{key}",
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            del env
            torch.cuda.empty_cache()

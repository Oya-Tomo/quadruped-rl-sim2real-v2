from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import genesis as gs
import torch

# Allow direct execution like: uv run src/eval/walking_terrain_mlp_ppo_eval.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.mlp import ActorNetwork as MLPActorNetwork
from src.model.mlp import ActorParams as MLPActorParams
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-s", "--steps", type=int, default=4096)
    parser.add_argument("-n", "--n_envs", type=int, default=1)
    parser.add_argument("-e", "--episode", type=int, default=1000)
    parser.add_argument("-H", "--headless", action="store_true")
    return parser.parse_args()


def _make_env(
    device: torch.device,
    n_envs: int,
    max_episode_length: int,
    headless: bool,
) -> WalkingTerrainEnv:
    sim_params = SimulationParams(
        n_envs=n_envs,
        frequency=50,
        substeps=4,
        max_episode_length=max_episode_length,
        shuffle_reset=False,
        begin_reset_episode_length=max_episode_length,
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
        traversability_measure_steps_threshold=500,
    )
    reward_params = RewardParams(
        reward_coef={
            "xy_velocity_tracking": 1.0,
            "yaw_velocity_tracking": 0.5,
            "command_tracking": 1.0,
            "curriculum_level": 0.5,
        },
        penalty_coef={
            "shoulder_height": -20.0,
            "body_stability": -0.05,
            "z_velocity": -0.2,
            "roll_pitch_velocity": -0.01,
            "dofs_power": -0.0001,
            "feet_velocity": -0.005,
            "feet_acceleration": -0.000001,
            "feet_contact_release": -0.05,
            "feet_slip": -0.05,
            "gait_pattern": -0.2,
            "raibert_heuristic": -1.0,
            "action_rate": -0.001,
            "action_smoothness": -0.001,
            "collision_prohibit": -8.0,
            "collision_avoidance": -4.0,
        },
        terminate_components=[
            "fall_down",
            "unsafe_xyz_velocity",
            "unsafe_rpy_velocity",
        ],
        truncate_components=[
            "timeout",
            "outside_field",
        ],
        shoulder_links=["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"],
        foot_links=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
        collision_reset_links=["base"],
        collision_prohibit_links=["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"],
        collision_avoidance_links=["FR_calf", "FL_calf", "RR_calf", "RL_calf"],
        collision_threshold=0.1,
        fall_down_threshold_radians=math.pi / 2,
    )
    test_params = TestParams(
        rendered_envs_idx=None,
        camera_pos=(-5.0, -5.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=60.0,
        show_viewer=not headless,
        enable_shortcut=True,
    )

    return WalkingTerrainEnv(
        sim_params=sim_params,
        robot_params=robot_params,
        control_params=control_params,
        command_params=command_params,
        terrain_curriculum_params=terrain_curriculum_params,
        reward_params=reward_params,
        field=PlaneField(),
        test_params=test_params,
        device=device,
    )


def _make_actor(env: WalkingTerrainEnv, device: torch.device) -> MLPActorNetwork:
    return MLPActorNetwork(
        MLPActorParams(
            state_dim=49,
            action_dim=env.dofs_num,
            hidden_dims=[512, 256, 128],
            state_keys=[
                ("observation", "base_xyz_velocity"),
                ("observation", "base_rpy_velocity"),
                ("observation", "dofs_position"),
                ("observation", "dofs_velocity"),
                ("observation", "projected_gravity"),
                ("observation", "command"),
                ("observation", "action"),
            ],
            action_key="action",
            state_independent_std=True,
        )
    ).to(device)


def main() -> None:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.n_envs <= 0:
        raise ValueError("--n_envs must be > 0")
    if args.episode <= 0:
        raise ValueError("--episode must be > 0")

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, seed=42, performance_mode=True, logging_level="error")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = _make_env(
        device=device,
        n_envs=args.n_envs,
        max_episode_length=args.episode,
        headless=args.headless,
    )
    actor_network = _make_actor(env=env, device=device)
    actor_module = actor_network.create_module().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor_state = checkpoint.get("actor_network")
    if actor_state is None:
        raise KeyError("Checkpoint does not contain actor_network")
    actor_network.load_state_dict(actor_state)
    actor_network.eval()

    rollout_steps = (args.steps + args.n_envs - 1) // args.n_envs

    try:
        with torch.no_grad():
            env.rollout(
                max_steps=rollout_steps,
                policy=actor_module,
                break_when_any_done=False,
                break_when_all_done=False,
            )
    finally:
        close_env = getattr(env, "close", None)
        if callable(close_env):
            close_env()


if __name__ == "__main__":
    main()

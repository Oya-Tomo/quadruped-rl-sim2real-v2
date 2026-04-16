from __future__ import annotations

import inspect
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Allow direct execution like: uv run src/train/walking_terrain_mlp_trpo.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import genesis as gs
import torch
from torchrl.record.loggers import WandbLogger

from src.algorithm.trpo import TRPOParams, TRPOTrainer
from src.model.mlp import ActorNetwork as MLPActorNetwork
from src.model.mlp import ActorParams as MLPActorParams
from src.model.mlp import CriticNetwork as MLPCriticNetwork
from src.model.mlp import CriticParams as MLPCriticParams
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


def _make_env(device: torch.device) -> WalkingTerrainEnv:
    sim_params = SimulationParams(
        n_envs=4096,
        frequency=50,
        substeps=4,
        max_episode_length=1200,
        shuffle_reset=True,
        begin_reset_episode_length=800,
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
            "body_stability": -0.2,
            "z_velocity": -0.5,
            "roll_pitch_velocity": -0.05,
            "dofs_velocity": -0.0002,
            "dofs_acceleration": -0.0000001,
            "dofs_power": -0.00002,
            "feet_velocity": -0.01,
            "feet_acceleration": -0.000002,
            "feet_slip": -0.5,
            "feet_contact_release": -2.0,
            "feet_soft_contact": -0.5,
            "gait_pattern": -1.0,
            "raibert_heuristic": -5.0,
            "jumping_motion": -1.5,
            "action_rate": -0.005,
            "action_smoothness": -0.005,
            "collision_prohibit": -10.0,
            "collision_avoidance": -5.0,
            "collision_dofs_limit": -8.0,
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
    )

    return WalkingTerrainEnv(
        sim_params=sim_params,
        robot_params=robot_params,
        control_params=control_params,
        command_params=command_params,
        terrain_curriculum_params=terrain_curriculum_params,
        reward_params=reward_params,
        field=PlaneField(),
        test_params=TestParams.default_to_train(),
        device=device,
    )


def main() -> None:
    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.gpu, seed=42, performance_mode=True, logging_level="error")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_name = "walking_terrain"
    exp_name = f"walking_terrain_mlp_trpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    env = _make_env(device=device)

    actor_network = MLPActorNetwork(
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
    critic_network = MLPCriticNetwork(
        MLPCriticParams(
            state_dim=50,
            hidden_dims=[512, 256, 128],
            value_dim=1,
            cost_dim=None,
            state_keys=[
                ("observation", "base_xyz_velocity"),
                ("observation", "base_rpy_velocity"),
                ("observation", "dofs_position"),
                ("observation", "dofs_velocity"),
                ("observation", "projected_gravity"),
                ("observation", "command"),
                ("observation", "action"),
                ("observation", "episode_progress"),
            ],
            value_key="value",
            value_cost_key=None,
        )
    ).to(device)

    trpo_params = TRPOParams(
        task_name=task_name,
        loops=20000,
        steps_per_batch=32,
        sub_batch_size=16384,
        critic_iters=10,
        checkpoint_interval=50,
        checkpoint_dir="checkpoints",
        gamma=0.99,
        lam=0.95,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        backtrack_steps=8,
        # backtrack_steps=10,
        backtrack_coeff=0.8,
        improve_ratio_threshold=0.1,
        entropy_loss_coeff=0.0001,
        critic_loss_coeff=1.0,
        value_loss_type="l2",
        use_rms=True,
        max_grad_norm=1.0,
        critic_lr=1e-4,
        weight_decay=1e-6,
        normalize_advantages=True,
        normalize_advantages_type="standard",
        rnn_mode=False,
        tbptt_steps=None,
    )

    logger = WandbLogger(
        exp_name=exp_name,
        project="quadruped-rl-sim2real",
        save_dir="wandb",
        config={
            "exp_name": exp_name,
            "task_name": task_name,
            "env": {
                "sim_params": asdict(env.sim_params),
                "robot_params": asdict(env.robot_params),
                "control_params": asdict(env.control_params),
                "command_params": asdict(env.command_params),
                "terrain_curriculum_params": asdict(env.terrain_curriculum_params),
                "reward_params": asdict(env.reward_params),
            },
            "trpo": asdict(trpo_params),
            "actor": asdict(actor_network.params),
            "critic": asdict(critic_network.params),
        },
    )

    trainer = TRPOTrainer(
        params=trpo_params,
        env=env,
        actor_network=actor_network,
        critic_network=critic_network,
        logger=logger,
    )

    try:
        trainer.train()
    finally:
        close_env = getattr(env, "close", None)
        if callable(close_env):
            close_env()
        close_logger = getattr(logger, "close", None)
        if callable(close_logger):
            close_logger()


if __name__ == "__main__":
    main()

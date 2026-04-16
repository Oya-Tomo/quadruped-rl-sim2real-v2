from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import genesis as gs
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, UnboundedContinuous, UnboundedDiscrete
from torchrl.envs import EnvBase

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

try:
    from src.task.field import Field
    from src.task.utils import bezier_3d, modified_cosine_similarity, rotation_matrix_2d
except ImportError:  # pragma: no cover
    from task.field import Field
    from task.utils import bezier_3d, modified_cosine_similarity, rotation_matrix_2d


@dataclass
class SimulationParams:
    n_envs: int
    frequency: int
    substeps: int
    max_episode_length: int
    shuffle_reset: bool
    begin_reset_episode_length: int


@dataclass
class RobotParams:
    urdf: str
    scale: float
    init_pos: list[float]
    init_quat: list[float]
    links_to_keep: list[str]


@dataclass
class ControlParams:
    dofs_names: list[str]
    dofs_origin: list[float]
    dofs_kp: list[float]
    dofs_kv: list[float]
    ema_alpha: float


@dataclass
class CommandParams:
    x_vel_min: float
    x_vel_max: float
    y_vel_min: float
    y_vel_max: float
    yaw_vel_min: float
    yaw_vel_max: float
    body_height_min: float
    body_height_max: float
    gait_period_min: float
    gait_period_max: float
    gait_phase_offset: list[float]
    resample_interval_avg: int


@dataclass
class RewardParams:
    reward_coef: dict[str, float] = field(default_factory=dict)
    penalty_coef: dict[str, float] = field(default_factory=dict)
    velocity_tracking_sigma: float = 0.1
    body_flip_sigm_coef: float = 20.0
    fall_down_threshold_radians: float = 1.5707963267948966
    feet_contact_threshold: float = 0.1
    feet_clearance_target: float = 0.08
    feet_contact_steps_threshold: int = 20
    raibert_feedback_coef: float = 0.05
    bezier_control_point_height: float = 0.1
    collision_dofs_limit_threshold: float = 0.05
    shoulder_links: list[str] = field(default_factory=list)
    foot_links: list[str] = field(default_factory=list)
    collision_reset_links: list[str] = field(default_factory=lambda: ["base"])
    collision_prohibit_links: list[str] = field(default_factory=list)
    collision_avoidance_links: list[str] = field(default_factory=list)
    collision_threshold: float = 0.5
    height_sampling_grid_x: list[float] = field(
        default_factory=lambda: [-0.2, 0.0, 0.2]
    )
    height_sampling_grid_y: list[float] = field(
        default_factory=lambda: [-0.1, 0.0, 0.1]
    )
    terminate_components: list[str] = field(default_factory=list)
    truncate_components: list[str] = field(default_factory=list)


@dataclass
class TerrainCurriculumParams:
    respawn_height: float
    respawn_range: list[tuple[float, float, float, float]]
    traversability_velocity_threshold: float = 0.5
    move_up_traversability: float = 0.7
    move_down_traversability: float = 0.3
    traversability_measure_steps_threshold: int = 1

    # Backward-compatible aliases for existing configs/tests.
    min_command_speed: float | None = None
    promotion_score_threshold: float | None = None
    demotion_score_threshold: float | None = None

    def __post_init__(self):
        if self.min_command_speed is not None:
            self.traversability_velocity_threshold = self.min_command_speed
        if self.promotion_score_threshold is not None:
            self.move_up_traversability = self.promotion_score_threshold
        if self.demotion_score_threshold is not None:
            self.move_down_traversability = self.demotion_score_threshold


@dataclass
class TestParams:
    rendered_envs_idx: list[int] | None
    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    camera_fov: float
    show_viewer: bool
    enable_shortcut: bool

    @classmethod
    def default_to_train(cls) -> Self:
        return TestParams(
            rendered_envs_idx=None,
            camera_pos=(0.0, 0.0, 0.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=60.0,
            show_viewer=False,
            enable_shortcut=False,
        )


class WalkingTerrainEnv(EnvBase):
    _batch_locked = True

    @dataclass
    class State:
        base_position: torch.Tensor
        base_quaternion: torch.Tensor
        base_rotation: torch.Tensor
        base_xyz_velocity: torch.Tensor
        base_rpy_velocity: torch.Tensor
        dofs_position: torch.Tensor
        dofs_velocity: torch.Tensor
        dofs_force: torch.Tensor
        projected_gravity: torch.Tensor
        links_position: torch.Tensor
        links_velocity: torch.Tensor
        links_acceleration: torch.Tensor
        links_external_force: torch.Tensor
        gait_phase: torch.Tensor
        shoulder_position: torch.Tensor
        shoulder_velocity: torch.Tensor
        feet_position: torch.Tensor
        feet_velocity: torch.Tensor
        feet_acceleration: torch.Tensor
        feet_contact: torch.Tensor
        feet_hold_steps: torch.Tensor
        feet_contact_accum_steps: torch.Tensor

        def copy_from(self, other: Self, envs_idx: torch.Tensor | None = None):
            if envs_idx is None:
                self.base_position[:] = other.base_position
                self.base_quaternion[:] = other.base_quaternion
                self.base_rotation[:] = other.base_rotation
                self.base_xyz_velocity[:] = other.base_xyz_velocity
                self.base_rpy_velocity[:] = other.base_rpy_velocity
                self.dofs_position[:] = other.dofs_position
                self.dofs_velocity[:] = other.dofs_velocity
                self.dofs_force[:] = other.dofs_force
                self.projected_gravity[:] = other.projected_gravity
                self.links_position[:] = other.links_position
                self.links_velocity[:] = other.links_velocity
                self.links_acceleration[:] = other.links_acceleration
                self.links_external_force[:] = other.links_external_force
                self.gait_phase[:] = other.gait_phase
                self.shoulder_position[:] = other.shoulder_position
                self.shoulder_velocity[:] = other.shoulder_velocity
                self.feet_position[:] = other.feet_position
                self.feet_velocity[:] = other.feet_velocity
                self.feet_acceleration[:] = other.feet_acceleration
                self.feet_contact[:] = other.feet_contact
                self.feet_hold_steps[:] = other.feet_hold_steps
                self.feet_contact_accum_steps[:] = other.feet_contact_accum_steps
            else:
                self.base_position[envs_idx] = other.base_position[envs_idx]
                self.base_quaternion[envs_idx] = other.base_quaternion[envs_idx]
                self.base_rotation[envs_idx] = other.base_rotation[envs_idx]
                self.base_xyz_velocity[envs_idx] = other.base_xyz_velocity[envs_idx]
                self.base_rpy_velocity[envs_idx] = other.base_rpy_velocity[envs_idx]
                self.dofs_position[envs_idx] = other.dofs_position[envs_idx]
                self.dofs_velocity[envs_idx] = other.dofs_velocity[envs_idx]
                self.dofs_force[envs_idx] = other.dofs_force[envs_idx]
                self.projected_gravity[envs_idx] = other.projected_gravity[envs_idx]
                self.links_position[envs_idx] = other.links_position[envs_idx]
                self.links_velocity[envs_idx] = other.links_velocity[envs_idx]
                self.links_acceleration[envs_idx] = other.links_acceleration[envs_idx]
                self.links_external_force[envs_idx] = other.links_external_force[
                    envs_idx
                ]
                self.gait_phase[envs_idx] = other.gait_phase[envs_idx]
                self.shoulder_position[envs_idx] = other.shoulder_position[envs_idx]
                self.shoulder_velocity[envs_idx] = other.shoulder_velocity[envs_idx]
                self.feet_position[envs_idx] = other.feet_position[envs_idx]
                self.feet_velocity[envs_idx] = other.feet_velocity[envs_idx]
                self.feet_acceleration[envs_idx] = other.feet_acceleration[envs_idx]
                self.feet_contact[envs_idx] = other.feet_contact[envs_idx]
                self.feet_hold_steps[envs_idx] = other.feet_hold_steps[envs_idx]
                self.feet_contact_accum_steps[envs_idx] = (
                    other.feet_contact_accum_steps[envs_idx]
                )

        def clone(self) -> Self:
            return WalkingTerrainEnv.State(
                base_position=self.base_position.detach().clone(),
                base_quaternion=self.base_quaternion.detach().clone(),
                base_rotation=self.base_rotation.detach().clone(),
                base_xyz_velocity=self.base_xyz_velocity.detach().clone(),
                base_rpy_velocity=self.base_rpy_velocity.detach().clone(),
                dofs_position=self.dofs_position.detach().clone(),
                dofs_velocity=self.dofs_velocity.detach().clone(),
                dofs_force=self.dofs_force.detach().clone(),
                projected_gravity=self.projected_gravity.detach().clone(),
                links_position=self.links_position.detach().clone(),
                links_velocity=self.links_velocity.detach().clone(),
                links_acceleration=self.links_acceleration.detach().clone(),
                links_external_force=self.links_external_force.detach().clone(),
                gait_phase=self.gait_phase.detach().clone(),
                shoulder_position=self.shoulder_position.detach().clone(),
                shoulder_velocity=self.shoulder_velocity.detach().clone(),
                feet_position=self.feet_position.detach().clone(),
                feet_velocity=self.feet_velocity.detach().clone(),
                feet_acceleration=self.feet_acceleration.detach().clone(),
                feet_contact=self.feet_contact.detach().clone(),
                feet_hold_steps=self.feet_hold_steps.detach().clone(),
                feet_contact_accum_steps=self.feet_contact_accum_steps.detach().clone(),
            )

    def __init__(
        self,
        sim_params: SimulationParams,
        robot_params: RobotParams,
        control_params: ControlParams,
        command_params: CommandParams,
        terrain_curriculum_params: TerrainCurriculumParams,
        field: Field,
        test_params: TestParams | None,
        device: torch.device,
        reward_params: RewardParams | None = None,
    ):
        super().__init__(device=device, batch_size=(sim_params.n_envs,))

        self.sim_params = sim_params
        self.robot_params = robot_params
        self.control_params = control_params
        self.command_params = command_params
        self.reward_params = (
            reward_params if reward_params is not None else RewardParams()
        )
        self.terrain_curriculum_params = terrain_curriculum_params
        self.test_params = (
            test_params if test_params is not None else TestParams.default_to_train()
        )
        self.dt = 1.0 / self.sim_params.frequency
        self.n_envs = self.sim_params.n_envs
        self.field_context = field.to(self.device)

        self.reward_fns = {
            "xy_velocity_tracking": self.calculate_reward_xy_velocity_tracking,
            "yaw_velocity_tracking": self.calculate_reward_yaw_velocity_tracking,
            "command_tracking": self.calculate_reward_command_tracking,
            "curriculum_level": self.calculate_reward_curriculum_level,
        }
        self.penalty_fns = {
            "body_height": self.calculate_penalty_body_height,
            "body_stability": self.calculate_penalty_body_stability,
            "body_flip": self.calculate_penalty_body_flip,
            "shoulder_height": self.calculate_penalty_shoulder_height,
            "z_velocity": self.calculate_penalty_z_velocity,
            "roll_pitch_velocity": self.calculate_penalty_roll_pitch_velocity,
            "dofs_velocity": self.calculate_penalty_dofs_velocity,
            "dofs_acceleration": self.calculate_penalty_dofs_acceleration,
            "dofs_power": self.calculate_penalty_dofs_power,
            "feet_velocity": self.calculate_penalty_feet_velocity,
            "feet_acceleration": self.calculate_penalty_feet_acceleration,
            "feet_slip": self.calculate_penalty_feet_slip,
            "feet_clearance": self.calculate_penalty_feet_clearance,
            "feet_contact_release": self.calculate_penalty_feet_contact_release,
            "feet_soft_contact": self.calculate_penalty_feet_soft_contact,
            "feet_contact_bias": self.calculate_penalty_feet_contact_bias,
            "gait_pattern": self.calculate_penalty_gait_pattern,
            "raibert_heuristic": self.calculate_penalty_raibert_heuristic,
            "jumping_motion": self.calculate_penalty_jumping_motion,
            "action_rate": self.calculate_penalty_action_rate,
            "action_smoothness": self.calculate_penalty_action_smoothness,
            "collision_prohibit": self.calculate_penalty_collision_prohibit,
            "collision_avoidance": self.calculate_penalty_collision_avoidance,
            "collision_dofs_limit": self.calculate_penalty_collision_dofs_limit,
        }
        self.reset_fns = {
            "fall_down": self.calculate_reset_fall_down,
            "unsafe_xyz_velocity": self.calculate_reset_unsafe_xyz_velocity,
            "unsafe_rpy_velocity": self.calculate_reset_unsafe_rpy_velocity,
            "outside_field": self.calculate_reset_outside_field,
            "timeout": self.calculate_reset_timeout,
            "collision_reset": self.calculate_reset_collision_reset,
        }

        unknown_reward_keys = set(self.reward_params.reward_coef.keys()) - set(
            self.reward_fns.keys()
        )
        if unknown_reward_keys:
            raise ValueError(f"Unknown reward keys: {sorted(unknown_reward_keys)}")
        unknown_penalty_keys = set(self.reward_params.penalty_coef.keys()) - set(
            self.penalty_fns.keys()
        )
        if unknown_penalty_keys:
            raise ValueError(f"Unknown penalty keys: {sorted(unknown_penalty_keys)}")
        unknown_terminate_keys = set(self.reward_params.terminate_components) - set(
            self.reset_fns.keys()
        )
        if unknown_terminate_keys:
            raise ValueError(
                f"Unknown terminate keys: {sorted(unknown_terminate_keys)}"
            )
        unknown_truncate_keys = set(self.reward_params.truncate_components) - set(
            self.reset_fns.keys()
        )
        if unknown_truncate_keys:
            raise ValueError(f"Unknown truncate keys: {sorted(unknown_truncate_keys)}")

        if not self.reward_params.reward_coef:
            self.reward_params.reward_coef = {
                key: 1.0 for key in self.reward_fns.keys()
            }

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=self.sim_params.substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=60,
                max_FPS=int(1.0 / self.dt),
                enable_default_keybinds=self.test_params.enable_shortcut,
                camera_lookat=self.test_params.camera_lookat,
                camera_pos=self.test_params.camera_pos,
                camera_fov=self.test_params.camera_fov,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=self.test_params.rendered_envs_idx,
            ),
            rigid_options=gs.options.RigidOptions(
                max_collision_pairs=20,
                batch_links_info=True,
                batch_joints_info=True,
                batch_dofs_info=True,
            ),
            show_viewer=self.test_params.show_viewer,
        )

        self.robot: RigidEntity = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=self.robot_params.urdf,
                scale=self.robot_params.scale,
                pos=self.robot_params.init_pos,
                quat=self.robot_params.init_quat,
                links_to_keep=self.robot_params.links_to_keep,
            ),
        )
        self.fields: list[RigidEntity] = [
            self.scene.add_entity(morph=morph)
            for morph in self.field_context.create_entities()
        ]
        for terrain in self.fields:
            terrain.set_friction(1.0)

        self.foot_links = (
            self.reward_params.foot_links
            if self.reward_params.foot_links
            else self.robot_params.links_to_keep
        )
        self.shoulder_links = self.reward_params.shoulder_links
        self.foot_links_idx = self._resolve_link_indices(self.foot_links)
        self.shoulder_links_idx = self._resolve_link_indices(self.shoulder_links)
        self.collision_reset_links_idx = self._resolve_link_indices(
            self.reward_params.collision_reset_links
        )
        self.collision_prohibit_links_idx = self._resolve_link_indices(
            self.reward_params.collision_prohibit_links
        )
        self.collision_avoidance_links_idx = self._resolve_link_indices(
            self.reward_params.collision_avoidance_links
        )
        self.height_sampling_points = torch.tensor(
            [
                [x, y]
                for x in self.reward_params.height_sampling_grid_x
                for y in self.reward_params.height_sampling_grid_y
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.command = torch.zeros((self.n_envs, 4), device=self.device)
        self.command_dist = torch.distributions.Uniform(
            low=torch.tensor(
                [
                    self.command_params.x_vel_min,
                    self.command_params.y_vel_min,
                    self.command_params.yaw_vel_min,
                    self.command_params.body_height_min,
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            high=torch.tensor(
                [
                    self.command_params.x_vel_max,
                    self.command_params.y_vel_max,
                    self.command_params.yaw_vel_max,
                    self.command_params.body_height_max,
                ],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.command_resample_prob = 1.0 / self.command_params.resample_interval_avg
        self.gait_period = torch.full(
            (self.n_envs,),
            (self.command_params.gait_period_min + self.command_params.gait_period_max)
            / 2.0,
            device=self.device,
            dtype=torch.float32,
        )
        self.gait_init_offset = torch.zeros(
            (self.n_envs,), dtype=torch.float32, device=self.device
        )
        self.gait_phase_offset = torch.tensor(
            self.command_params.gait_phase_offset,
            device=self.device,
            dtype=torch.float32,
        )

        self.episode_steps = torch.zeros(
            self.n_envs, dtype=torch.int32, device=self.device
        )
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        self.base_init_position = torch.tensor(
            self.robot_params.init_pos, device=self.device, dtype=torch.float32
        )
        self.base_init_quaternion = torch.tensor(
            self.robot_params.init_quat, device=self.device, dtype=torch.float32
        )
        self.base_init_quaternion_inv = gs.inv_quat(self.base_init_quaternion)

        self.dofs_num = len(self.control_params.dofs_names)
        self.dofs_idx = [
            self.robot.get_joint(name).dof_start
            for name in self.control_params.dofs_names
        ]
        self.dofs_lower_limit = torch.tensor(
            [
                self.robot.get_joint(name).dofs_limit[0][0]
                for name in self.control_params.dofs_names
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.dofs_upper_limit = torch.tensor(
            [
                self.robot.get_joint(name).dofs_limit[0][1]
                for name in self.control_params.dofs_names
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.dofs_origin = torch.tensor(
            self.control_params.dofs_origin, device=self.device, dtype=torch.float32
        )
        self.dofs_kp = torch.tensor(
            self.control_params.dofs_kp, device=self.device, dtype=torch.float32
        )
        self.dofs_kv = torch.tensor(
            self.control_params.dofs_kv, device=self.device, dtype=torch.float32
        )
        self.dofs_target = torch.zeros(
            (self.n_envs, self.dofs_num), device=self.device, dtype=torch.float32
        )
        self.action_buffer = torch.zeros(
            (self.n_envs, 3, self.dofs_num), device=self.device, dtype=torch.float32
        )
        self.action_applied = self.action_buffer[:, 0, :]

        self.curriculum_level_num = len(self.terrain_curriculum_params.respawn_range)
        if self.curriculum_level_num <= 0:
            raise ValueError("respawn_range must not be empty")
        self.curriculum_level = torch.zeros(
            self.n_envs, dtype=torch.int32, device=self.device
        )
        self.respawn_position = torch.zeros((self.n_envs, 3), device=self.device)
        self.respawn_range = torch.tensor(
            self.terrain_curriculum_params.respawn_range,
            dtype=torch.float32,
            device=self.device,
        )
        self.traversability_total = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.float32
        )
        self.traversability_count_steps = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.int32
        )

        self.state = self.State(
            base_position=torch.zeros((self.n_envs, 3), device=self.device),
            base_quaternion=torch.zeros((self.n_envs, 4), device=self.device),
            base_rotation=torch.zeros((self.n_envs, 3), device=self.device),
            base_xyz_velocity=torch.zeros((self.n_envs, 3), device=self.device),
            base_rpy_velocity=torch.zeros((self.n_envs, 3), device=self.device),
            dofs_position=torch.zeros((self.n_envs, self.dofs_num), device=self.device),
            dofs_velocity=torch.zeros((self.n_envs, self.dofs_num), device=self.device),
            dofs_force=torch.zeros((self.n_envs, self.dofs_num), device=self.device),
            projected_gravity=torch.zeros((self.n_envs, 3), device=self.device),
            links_position=torch.zeros(
                (self.n_envs, self.robot.n_links, 3), device=self.device
            ),
            links_velocity=torch.zeros(
                (self.n_envs, self.robot.n_links, 3), device=self.device
            ),
            links_acceleration=torch.zeros(
                (self.n_envs, self.robot.n_links, 3), device=self.device
            ),
            links_external_force=torch.zeros(
                (self.n_envs, self.robot.n_links, 3), device=self.device
            ),
            gait_phase=torch.zeros((self.n_envs, 4), device=self.device),
            shoulder_position=torch.zeros(
                (self.n_envs, len(self.shoulder_links_idx), 3), device=self.device
            ),
            shoulder_velocity=torch.zeros(
                (self.n_envs, len(self.shoulder_links_idx), 3), device=self.device
            ),
            feet_position=torch.zeros(
                (self.n_envs, len(self.foot_links_idx), 3), device=self.device
            ),
            feet_velocity=torch.zeros(
                (self.n_envs, len(self.foot_links_idx), 3), device=self.device
            ),
            feet_acceleration=torch.zeros(
                (self.n_envs, len(self.foot_links_idx), 3), device=self.device
            ),
            feet_contact=torch.zeros(
                (self.n_envs, len(self.foot_links_idx)),
                device=self.device,
                dtype=torch.bool,
            ),
            feet_hold_steps=torch.zeros(
                (self.n_envs, len(self.foot_links_idx)),
                device=self.device,
                dtype=torch.int32,
            ),
            feet_contact_accum_steps=torch.zeros(
                (self.n_envs, len(self.foot_links_idx)),
                device=self.device,
                dtype=torch.int32,
            ),
        )
        self.state_prev = self.state.clone()

        self._make_spec()
        self.scene.build(n_envs=self.n_envs)

    def _make_spec(self):
        reward_components_spec = {
            key: UnboundedContinuous(shape=(1,), dtype=torch.float32)
            for key in self.reward_params.reward_coef.keys()
        }
        reward_components_spec["total"] = UnboundedContinuous(
            shape=(1,), dtype=torch.float32
        )
        penalty_components_spec = {
            key: UnboundedContinuous(shape=(1,), dtype=torch.float32)
            for key in self.reward_params.penalty_coef.keys()
        }
        penalty_components_spec["total"] = UnboundedContinuous(
            shape=(1,), dtype=torch.float32
        )
        self.reward_spec_unbatched = Composite(
            {
                "reward": UnboundedContinuous(shape=(1,), dtype=torch.float32),
                "info": Composite(
                    {
                        "reward_components": Composite(reward_components_spec),
                        "penalty_components": Composite(penalty_components_spec),
                    }
                ),
            }
        )
        self.observation_spec_unbatched = Composite(
            {
                "observation": Composite(
                    {
                        "base_xyz_velocity": UnboundedContinuous(
                            shape=(3,), dtype=torch.float32
                        ),
                        "base_rpy_velocity": UnboundedContinuous(
                            shape=(3,), dtype=torch.float32
                        ),
                        "dofs_position": UnboundedContinuous(
                            shape=(self.dofs_num,), dtype=torch.float32
                        ),
                        "dofs_velocity": UnboundedContinuous(
                            shape=(self.dofs_num,), dtype=torch.float32
                        ),
                        "projected_gravity": UnboundedContinuous(
                            shape=(3,), dtype=torch.float32
                        ),
                        "command": UnboundedContinuous(shape=(4,), dtype=torch.float32),
                        "action": UnboundedContinuous(
                            shape=(self.dofs_num,), dtype=torch.float32
                        ),
                        "gait_phase_sin": UnboundedContinuous(
                            shape=(4,), dtype=torch.float32
                        ),
                        "gait_phase_cos": UnboundedContinuous(
                            shape=(4,), dtype=torch.float32
                        ),
                        "episode_progress": UnboundedContinuous(
                            shape=(1,), dtype=torch.float32
                        ),
                        "curriculum_level": UnboundedDiscrete(
                            shape=(1,), dtype=torch.int32
                        ),
                    }
                )
            }
        )
        self.action_spec_unbatched = UnboundedContinuous(
            shape=(self.dofs_num,), dtype=torch.float32
        )
        self.done_spec_unbatched = Composite(
            {
                "done": Binary(shape=(1,), n=1, dtype=torch.bool),
                "terminated": Binary(shape=(1,), n=1, dtype=torch.bool),
                "truncated": Binary(shape=(1,), n=1, dtype=torch.bool),
            }
        )

    def _resolve_link_indices(self, names: list[str]) -> list[int]:
        indices: list[int] = []
        for name in names:
            try:
                indices.append(self.robot.get_link(name).idx_local)
            except Exception:
                continue
        return indices

    def _set_seed(self, seed):
        gs.set_random_seed(seed)

    def _destroy_scene(self):
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        scene.destroy()
        self.scene = None

    def close(self, *, raise_if_closed: bool = True):
        self._destroy_scene()
        super().close(raise_if_closed=raise_if_closed)

    def __del__(self):
        try:
            self._destroy_scene()
        except Exception:
            pass

    def _reset(self, td: TensorDict | None) -> TensorDict:
        if td is None:
            reset_mask = torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            ).flatten()
        else:
            reset_mask = td.get("_reset").flatten()

        envs_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
        self.reset_idx(envs_idx=envs_idx)
        return TensorDict(
            {"observation": self.extract_observation()},
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        self.apply_action(tensordict["action"])
        self.simulate_step()
        self.update_state()

        terminate_components = self.extract_terminate()
        terminated = self.combine_reset_components(terminate_components)
        truncate_components = self.extract_truncate()
        truncated = self.combine_reset_components(truncate_components)
        done = terminated | truncated

        reward_components = self.calculate_reward_components(
            terminated=terminated,
            truncated=truncated,
        )
        task_reward_sum, penalty_reward_sum = self._sum_task_and_penalty_rewards(
            reward_components
        )
        reward_info_components = TensorDict(
            {
                key: reward_components[key]
                for key in self.reward_params.reward_coef.keys()
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        reward_info_components["total"] = task_reward_sum.unsqueeze(-1)

        penalty_info_components = TensorDict(
            {
                key: reward_components[key]
                for key in self.reward_params.penalty_coef.keys()
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        penalty_info_components["total"] = penalty_reward_sum.unsqueeze(-1)

        reward = self.combine_reward_components(reward_components)
        return TensorDict(
            {
                "observation": self.extract_observation(),
                "reward": reward.unsqueeze(-1),
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
                "info": TensorDict(
                    {
                        "reward_components": reward_info_components,
                        "penalty_components": penalty_info_components,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def reset_idx(self, envs_idx: torch.Tensor):
        if envs_idx.numel() == 0:
            return
        self.update_curriculum(envs_idx=envs_idx)
        self.sample_respawn_position(envs_idx=envs_idx)
        self.sample_command(envs_idx=envs_idx)
        self.reset_scene_idx(envs_idx=envs_idx)
        self.reset_state_idx(envs_idx=envs_idx)

    def reset_scene_idx(self, envs_idx: torch.Tensor):
        self.scene.reset(envs_idx=envs_idx)
        self.robot.set_pos(
            pos=self.respawn_position[envs_idx],
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self.robot.set_quat(
            quat=self.base_init_quaternion.expand(envs_idx.numel(), -1),
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self.robot.set_dofs_position(
            position=self.dofs_origin.expand(envs_idx.numel(), -1),
            dofs_idx_local=self.dofs_idx,
            envs_idx=envs_idx,
            zero_velocity=True,
        )

    def reset_state_idx(self, envs_idx: torch.Tensor):
        self.episode_steps[envs_idx] = 0
        self.traversability_total[envs_idx] = 0.0
        self.traversability_count_steps[envs_idx] = 0
        self.action_buffer[envs_idx] = 0.0

        self.state.base_position[envs_idx] = self.robot.get_pos(envs_idx=envs_idx)
        self.state.base_quaternion[envs_idx] = gs.transform_quat_by_quat(
            self.base_init_quaternion_inv.expand(envs_idx.numel(), -1),
            self.robot.get_quat(envs_idx=envs_idx),
        )
        self.state.base_rotation[envs_idx] = gs.quat_to_xyz(
            self.state.base_quaternion[envs_idx], rpy=True, degrees=False
        )
        inv_base_quat = gs.inv_quat(self.state.base_quaternion[envs_idx])
        self.state.base_xyz_velocity[envs_idx] = gs.transform_by_quat(
            self.robot.get_vel(envs_idx=envs_idx), inv_base_quat
        )
        self.state.base_rpy_velocity[envs_idx] = gs.transform_by_quat(
            self.robot.get_ang(envs_idx=envs_idx), inv_base_quat
        )
        self.state.dofs_position[envs_idx] = self.robot.get_dofs_position(
            dofs_idx_local=self.dofs_idx,
            envs_idx=envs_idx,
        )
        self.state.dofs_velocity[envs_idx] = self.robot.get_dofs_velocity(
            dofs_idx_local=self.dofs_idx,
            envs_idx=envs_idx,
        )
        self.state.dofs_force[envs_idx] = self.robot.get_dofs_force(
            dofs_idx_local=self.dofs_idx,
            envs_idx=envs_idx,
        )
        self.state.projected_gravity[envs_idx] = gs.transform_by_quat(
            self.global_gravity.expand(envs_idx.numel(), -1), inv_base_quat
        )
        self.state.links_position[envs_idx] = self.robot.get_links_pos(
            envs_idx=envs_idx
        )
        self.state.links_velocity[envs_idx] = self.robot.get_links_vel(
            envs_idx=envs_idx
        )
        self.state.links_acceleration[envs_idx] = self.robot.get_links_acc(
            envs_idx=envs_idx
        )
        self.state.links_external_force[envs_idx] = (
            self.robot.get_links_net_contact_force(envs_idx=envs_idx)
        )

        self.dofs_target[envs_idx] = self.state.dofs_position[envs_idx]
        self.state.gait_phase[envs_idx] = 0.0
        if self.shoulder_links_idx:
            self.state.shoulder_position[envs_idx] = self.state.links_position[
                envs_idx, :
            ][:, self.shoulder_links_idx]
            self.state.shoulder_velocity[envs_idx] = self.state.links_velocity[
                envs_idx, :
            ][:, self.shoulder_links_idx]
        if self.foot_links_idx:
            self.state.feet_position[envs_idx] = self.state.links_position[envs_idx, :][
                :, self.foot_links_idx
            ]
            self.state.feet_velocity[envs_idx] = self.state.links_velocity[envs_idx, :][
                :, self.foot_links_idx
            ]
            self.state.feet_acceleration[envs_idx] = self.state.links_acceleration[
                envs_idx, :
            ][:, self.foot_links_idx]
            self.state.feet_contact[envs_idx] = (
                self.state.links_external_force[envs_idx, :][
                    :, self.foot_links_idx
                ].norm(dim=-1)
                > self.reward_params.feet_contact_threshold
            )
            self.state.feet_hold_steps[envs_idx] = 0
            self.state.feet_contact_accum_steps[envs_idx] = 0
        self.state_prev.copy_from(self.state, envs_idx)

    def sample_command(self, envs_idx: torch.Tensor):
        self.command[envs_idx] = self.command_dist.sample((envs_idx.numel(),))
        self.gait_period[envs_idx] = (
            torch.rand((envs_idx.numel(),), device=self.device)
            * (
                self.command_params.gait_period_max
                - self.command_params.gait_period_min
            )
            + self.command_params.gait_period_min
        )
        self.gait_init_offset[envs_idx] = torch.rand(
            (envs_idx.numel(),), device=self.device
        ) * (2.0 * math.pi)

    def sample_respawn_position(self, envs_idx: torch.Tensor):
        levels = self.curriculum_level[envs_idx]
        respawn = self.respawn_range[levels]
        respawn_x = respawn[:, 0] + torch.rand(
            (envs_idx.numel(),), device=self.device
        ) * (respawn[:, 1] - respawn[:, 0])
        respawn_y = respawn[:, 2] + torch.rand(
            (envs_idx.numel(),), device=self.device
        ) * (respawn[:, 3] - respawn[:, 2])
        respawn_z = (
            self.field_context.get_height(torch.stack([respawn_x, respawn_y], dim=-1))
            + self.terrain_curriculum_params.respawn_height
        )
        self.respawn_position[envs_idx] = torch.stack(
            [respawn_x, respawn_y, respawn_z], dim=-1
        )

    def update_curriculum(self, envs_idx: torch.Tensor):
        survived = self.episode_steps[envs_idx] >= self.sim_params.max_episode_length
        traversability_mean = self.traversability_total[envs_idx] / (
            self.traversability_count_steps[envs_idx].to(torch.float32) + 1e-8
        )

        leveled_up = (
            (
                traversability_mean
                >= self.terrain_curriculum_params.move_up_traversability
            )
            & survived
            & (
                self.traversability_count_steps[envs_idx]
                >= self.terrain_curriculum_params.traversability_measure_steps_threshold
            )
        )
        leveled_down = (
            (
                traversability_mean
                <= self.terrain_curriculum_params.move_down_traversability
            )
            & (self.traversability_count_steps[envs_idx] > 0)
        ) | survived.logical_not()

        self.curriculum_level[envs_idx] += leveled_up.to(torch.int32)
        self.curriculum_level[envs_idx] -= leveled_down.to(torch.int32)
        max_level = self.curriculum_level_num - 1
        over_max = envs_idx[self.curriculum_level[envs_idx] > max_level]
        if over_max.numel() > 0:
            self.curriculum_level[over_max] = torch.randint(
                0,
                max_level + 1,
                (over_max.numel(),),
                device=self.device,
                dtype=torch.int32,
            )
        self.curriculum_level[envs_idx] = self.curriculum_level[envs_idx].clamp(
            0, max_level
        )

    def apply_action(self, action: torch.Tensor):
        self.action_buffer[:, 1:, :] = self.action_buffer[:, :-1, :].clone()
        self.action_buffer[:, 0, :] = torch.clamp(action, -100.0, 100.0)
        action_applied = self.action_buffer[:, 0, :]
        self.dofs_target[:] = (
            self.control_params.ema_alpha * (action_applied + self.dofs_origin)
            + (1.0 - self.control_params.ema_alpha) * self.dofs_target
        )

    def calculate_nonlinear_dofs_target(self) -> torch.Tensor:
        return self.dofs_target

    def simulate_step(self):
        self.robot.control_dofs_position(
            self.calculate_nonlinear_dofs_target(),
            dofs_idx_local=self.dofs_idx,
        )
        self.scene.step()

    def update_state(self):
        self.state_prev.copy_from(self.state)
        self.state.base_position[:] = self.robot.get_pos()
        self.state.base_quaternion[:] = gs.transform_quat_by_quat(
            self.base_init_quaternion_inv.expand(self.n_envs, -1),
            self.robot.get_quat(),
        )
        self.state.base_rotation[:] = gs.quat_to_xyz(
            self.state.base_quaternion, rpy=True, degrees=False
        )
        inv_base_quat = gs.inv_quat(self.state.base_quaternion)
        self.state.base_xyz_velocity[:] = gs.transform_by_quat(
            self.robot.get_vel(), inv_base_quat
        )
        self.state.base_rpy_velocity[:] = gs.transform_by_quat(
            self.robot.get_ang(), inv_base_quat
        )
        self.state.dofs_position[:] = self.robot.get_dofs_position(
            dofs_idx_local=self.dofs_idx
        )
        self.state.dofs_velocity[:] = self.robot.get_dofs_velocity(
            dofs_idx_local=self.dofs_idx
        )
        self.state.dofs_force[:] = self.robot.get_dofs_force(
            dofs_idx_local=self.dofs_idx
        )
        self.state.projected_gravity[:] = gs.transform_by_quat(
            self.global_gravity, inv_base_quat
        )
        self.state.links_position[:] = self.robot.get_links_pos()
        self.state.links_velocity[:] = self.robot.get_links_vel()
        self.state.links_acceleration[:] = self.robot.get_links_acc()
        self.state.links_external_force[:] = self.robot.get_links_net_contact_force()

        self.episode_steps += 1

        self.state.gait_phase[:] = torch.fmod(
            (
                2.0
                * math.pi
                * self.episode_steps.to(torch.float32)
                * self.dt
                / self.gait_period
            ).unsqueeze(-1)
            + self.gait_init_offset.unsqueeze(-1)
            + self.gait_phase_offset.unsqueeze(0),
            2.0 * math.pi,
        )
        if self.shoulder_links_idx:
            self.state.shoulder_position[:] = self.state.links_position[
                :, self.shoulder_links_idx
            ]
            self.state.shoulder_velocity[:] = self.state.links_velocity[
                :, self.shoulder_links_idx
            ]
        if self.foot_links_idx:
            self.state.feet_position[:] = self.state.links_position[
                :, self.foot_links_idx
            ]
            self.state.feet_velocity[:] = self.state.links_velocity[
                :, self.foot_links_idx
            ]
            self.state.feet_acceleration[:] = self.state.links_acceleration[
                :, self.foot_links_idx
            ]
            self.state.feet_contact[:] = (
                self.state.links_external_force[:, self.foot_links_idx].norm(dim=-1)
                > self.reward_params.feet_contact_threshold
            )
            self.state.feet_hold_steps[:] = (self.state_prev.feet_hold_steps + 1) * (
                self.state.feet_contact == self.state_prev.feet_contact
            ).to(torch.int32)
            self.state.feet_contact_accum_steps[:] = (
                self.state_prev.feet_contact_accum_steps
                + self.state.feet_contact.to(torch.int32)
            )

        traversability_mask = (
            self.command[:, :2].norm(dim=-1)
            > self.terrain_curriculum_params.traversability_velocity_threshold
        )
        traversability = self.calculate_traversability() * traversability_mask.to(
            torch.float32
        )
        self.traversability_total += traversability
        self.traversability_count_steps += traversability_mask.to(torch.int32)

        command_resample_envs_idx = torch.nonzero(
            torch.rand((self.n_envs,), device=self.device) < self.command_resample_prob,
            as_tuple=False,
        ).flatten()
        self.sample_command(command_resample_envs_idx)

    def combine_reset_components(self, reset_components: TensorDict) -> torch.Tensor:
        reset = torch.zeros((self.n_envs,), device=self.device, dtype=torch.bool)
        for component in reset_components.keys():
            reset = reset | reset_components[component]
        return reset

    def extract_terminate(self) -> TensorDict:
        return TensorDict(
            {
                key: self.reset_fns[key]()
                for key in self.reward_params.terminate_components
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def extract_truncate(self) -> TensorDict:
        return TensorDict(
            {
                key: self.reset_fns[key]()
                for key in self.reward_params.truncate_components
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def calculate_traversability(self) -> torch.Tensor:
        command_xyz = torch.cat(
            [
                self.command[:, :2],
                self.command[:, 2:3],
            ],
            dim=-1,
        )
        body_velocity = torch.cat(
            [
                self.state.base_xyz_velocity[:, :2],
                self.state.base_rpy_velocity[:, 2:3],
            ],
            dim=-1,
        )
        return modified_cosine_similarity(body_velocity, command_xyz, dim=1)

    def calculate_tracking_xy_score(self) -> torch.Tensor:
        xy_error = (
            (self.command[:, :2] - self.state.base_xyz_velocity[:, :2])
            .square()
            .sum(dim=-1)
        )
        return torch.exp(-xy_error / self.reward_params.velocity_tracking_sigma)

    def calculate_tracking_yaw_score(self) -> torch.Tensor:
        yaw_error = (self.command[:, 2] - self.state.base_rpy_velocity[:, 2]).square()
        return torch.exp(-yaw_error / self.reward_params.velocity_tracking_sigma)

    def calculate_reward_components(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> TensorDict:
        reward_components = TensorDict(
            {}, batch_size=self.batch_size, device=self.device
        )

        for reward_name, coef in self.reward_params.reward_coef.items():
            reward_fn = self.reward_fns[reward_name]
            reward_components[reward_name] = (
                reward_fn(terminated=terminated, truncated=truncated).unsqueeze(-1)
                * coef
            )
        for penalty_name, coef in self.reward_params.penalty_coef.items():
            penalty_fn = self.penalty_fns[penalty_name]
            reward_components[penalty_name] = (
                penalty_fn(terminated=terminated, truncated=truncated).unsqueeze(-1)
                * coef
            )

        return reward_components

    def _sum_task_and_penalty_rewards(
        self, reward_components: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_reward = torch.zeros((self.n_envs,), device=self.device)
        penalty_reward = torch.zeros((self.n_envs,), device=self.device)

        for key, reward_value in reward_components.items():
            value = reward_value.squeeze(-1)
            if key in self.reward_params.penalty_coef:
                penalty_reward += value
            else:
                task_reward += value

        return task_reward, penalty_reward

    def combine_reward_components(self, reward_components: TensorDict) -> torch.Tensor:
        task_reward, penalty_reward = self._sum_task_and_penalty_rewards(
            reward_components
        )
        return task_reward * torch.exp(penalty_reward * 0.15)

    def calculate_reward_xy_velocity_tracking(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.calculate_tracking_xy_score()

    def calculate_reward_yaw_velocity_tracking(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.calculate_tracking_yaw_score()

    def calculate_reward_command_tracking(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        return self.calculate_reward_xy_velocity_tracking(
            terminated=terminated,
            truncated=truncated,
        ) * self.calculate_reward_yaw_velocity_tracking(
            terminated=terminated,
            truncated=truncated,
        )

    def calculate_reward_curriculum_level(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.curriculum_level.to(torch.float32) / float(
            max(1, self.curriculum_level_num - 1)
        )

    def calculate_penalty_body_height(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        yaw_rotate_mat = rotation_matrix_2d(self.state.base_rotation[:, 2])
        height_sample_points = torch.einsum(
            "bij,pj->bpi",
            yaw_rotate_mat,
            self.height_sampling_points,
        ) + self.state.base_position[:, :2].unsqueeze(-2)
        height_target = (
            self.field_context.get_height(height_sample_points).mean(dim=-1)
            + self.command[:, 3]
        )
        return (height_target - self.state.base_position[:, 2]).abs()

    def calculate_penalty_body_stability(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.state.projected_gravity[:, 2].clamp(-1.0, 1.0).asin().abs()

    def calculate_penalty_body_flip(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return (
            self.state.projected_gravity[:, 2] * self.reward_params.body_flip_sigm_coef
        ).sigmoid()

    def calculate_penalty_shoulder_height(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.shoulder_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        height_sample_points = self.state.shoulder_position[:, :, :2]
        heights = self.field_context.get_height(height_sample_points)
        height_min = (heights.min(dim=-1).values + self.command[:, 3]).unsqueeze(-1)
        height_max = (heights.max(dim=-1).values + self.command[:, 3]).unsqueeze(-1)
        diff_min = (height_min - self.state.shoulder_position[:, :, 2]).clamp_min(0.0)
        diff_max = (self.state.shoulder_position[:, :, 2] - height_max).clamp_min(0.0)
        return (diff_min + diff_max).mean(dim=-1)

    def calculate_penalty_z_velocity(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.state.base_xyz_velocity[:, 2].square()

    def calculate_penalty_roll_pitch_velocity(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.state.base_rpy_velocity[:, :2].square().sum(dim=-1)

    def calculate_penalty_dofs_velocity(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return self.state.dofs_velocity.square().sum(dim=-1)

    def calculate_penalty_dofs_acceleration(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return (
            ((self.state.dofs_velocity - self.state_prev.dofs_velocity) / self.dt)
            .square()
            .sum(dim=-1)
        )

    def calculate_penalty_dofs_power(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return (self.state.dofs_force * self.state.dofs_velocity).abs().sum(dim=-1)

    def calculate_penalty_feet_velocity(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        return self.state.feet_velocity.square().sum(dim=-1).sum(dim=-1)

    def calculate_penalty_feet_acceleration(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        return self.state.feet_acceleration.square().sum(dim=-1).sum(dim=-1)

    def calculate_penalty_feet_slip(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        feet_contact_phase = (
            self.state.gait_phase.sin() >= 0.0
        ) | self.state.feet_contact
        feet_velocity = self.state.feet_velocity.norm(dim=-1)
        return (feet_velocity * feet_contact_phase.to(torch.float32)).sum(dim=-1)

    def calculate_penalty_feet_clearance(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        feet_height = self.state.feet_position[:, :, 2] - self.field_context.get_height(
            self.state.feet_position[:, :, :2]
        )
        return (
            self.state.feet_velocity[:, :, :2].norm(dim=-1)
            * (self.reward_params.feet_clearance_target - feet_height).square()
        ).sum(dim=-1)

    def calculate_penalty_feet_contact_release(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        contact_penalty = (
            (-self.state.gait_phase.cos() + 1.0)
            / 2.0
            * (self.state.feet_contact & self.state_prev.feet_contact.logical_not()).to(
                torch.float32
            )
        )
        release_penalty = (
            (self.state.gait_phase.cos() + 1.0)
            / 2.0
            * (self.state.feet_contact.logical_not() & self.state_prev.feet_contact).to(
                torch.float32
            )
        )
        return (contact_penalty + release_penalty).sum(dim=-1)

    def calculate_penalty_feet_soft_contact(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        penalty = (
            self.state.feet_contact.to(torch.float32)
            * self.state_prev.feet_velocity[:, :, 2].abs()
        )
        return penalty.sum(dim=-1)

    def calculate_penalty_feet_contact_bias(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        return (
            self.state.feet_contact_accum_steps.max(-1).values
            - self.state.feet_contact_accum_steps.min(-1).values
            - self.reward_params.feet_contact_steps_threshold * 2
        ).to(torch.float32).clamp_min(0.0) * self.dt

    def calculate_penalty_gait_pattern(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        return (
            (
                (self.state.gait_phase.sin() >= 0.0)
                & self.state.feet_contact.logical_not()
            )
            .to(torch.float32)
            .sum(dim=-1)
        )

    def calculate_penalty_raibert_heuristic(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx or not self.shoulder_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)

        global_velocity_target = torch.zeros(
            (self.n_envs, 3), device=self.device, dtype=torch.float32
        )
        global_velocity_target[:, 0:2] = self.command[:, 0:2]
        global_velocity_target = gs.transform_by_quat(
            global_velocity_target,
            self.state.base_quaternion,
        )
        shoulder_radius = self.state.shoulder_position[
            :, :, :2
        ] - self.state.base_position[:, :2].unsqueeze(-2)
        shoulder_tangent = torch.einsum(
            "ij,blj->bli",
            rotation_matrix_2d(torch.tensor(math.pi / 2.0, device=self.device)),
            shoulder_radius,
        )
        shoulder_velocity_target = global_velocity_target[:, :2].unsqueeze(
            -2
        ) + shoulder_tangent * self.command[:, 2].unsqueeze(-1).unsqueeze(-1)

        contact_duration = self.gait_period.unsqueeze(-1).unsqueeze(-1) / 2.0
        feedback = self.reward_params.raibert_feedback_coef * (
            self.state.shoulder_velocity[:, :, :2] - shoulder_velocity_target
        )
        contact_target = (
            self.state.shoulder_position[:, :, :2]
            + (contact_duration * shoulder_velocity_target / 2.0)
            + feedback
        )
        release_target = (
            self.state.shoulder_position[:, :, :2]
            - (contact_duration * shoulder_velocity_target / 2.0)
            + feedback
        )
        contact_point = torch.cat(
            [
                contact_target,
                self.field_context.get_height(contact_target[:, :, :2]).unsqueeze(-1),
            ],
            dim=-1,
        )
        contact_control_point = contact_point + torch.tensor(
            [0.0, 0.0, self.reward_params.bezier_control_point_height],
            device=self.device,
        )
        release_point = torch.cat(
            [
                release_target,
                self.field_context.get_height(release_target[:, :, :2]).unsqueeze(-1),
            ],
            dim=-1,
        )
        release_control_point = release_point + torch.tensor(
            [0.0, 0.0, self.reward_params.bezier_control_point_height],
            device=self.device,
        )

        stance_phase_rate = self.state.gait_phase / math.pi
        stance_position = (
            1.0 - stance_phase_rate.unsqueeze(-1)
        ) * contact_target + stance_phase_rate.unsqueeze(-1) * release_target
        stance_point = torch.cat(
            [
                stance_position,
                self.field_context.get_height(stance_position[:, :, :2]).unsqueeze(-1),
            ],
            dim=-1,
        )
        swing_phase_rate = (self.state.gait_phase - math.pi) / math.pi
        swing_point = bezier_3d(
            release_point,
            release_control_point,
            contact_point,
            contact_control_point,
            swing_phase_rate.unsqueeze(-1),
        )
        target_point = torch.where(
            self.state.gait_phase.unsqueeze(-1) < math.pi,
            stance_point,
            swing_point,
        )

        self.scene.clear_debug_objects()
        self.scene.draw_debug_spheres(
            target_point.view(-1, 3), color=(0.0, 1.0, 0.0), radius=0.02
        )

        return (
            (self.state.feet_position - target_point).square().sum(dim=-1).sum(dim=-1)
        )

    def calculate_penalty_jumping_motion(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        if not self.foot_links_idx:
            return torch.zeros((self.n_envs,), device=self.device)
        feet_release_num = (~self.state.feet_contact).to(torch.float32).sum(-1)
        return (feet_release_num > 2.5).to(torch.float32)

    def calculate_penalty_action_rate(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return (
            (self.action_buffer[:, 0, :] - self.action_buffer[:, 1, :])
            .square()
            .sum(dim=-1)
        )

    def calculate_penalty_action_smoothness(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        return (
            (
                self.action_buffer[:, 0, :]
                - 2 * self.action_buffer[:, 1, :]
                + self.action_buffer[:, 2, :]
            )
            .square()
            .sum(dim=-1)
        )

    def _calculate_collision_contacts(self, links_idx: list[int]) -> torch.Tensor:
        if not links_idx:
            return torch.zeros((self.n_envs, 0), dtype=torch.bool, device=self.device)
        return (
            self.state.links_external_force[:, links_idx].norm(dim=-1)
            > self.reward_params.collision_threshold
        )

    def calculate_penalty_collision_prohibit(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        contacts = self._calculate_collision_contacts(self.collision_prohibit_links_idx)
        if contacts.shape[1] == 0:
            return torch.zeros((self.n_envs,), device=self.device)
        # Strong penalty: count shoulder-body contacts.
        return contacts.to(torch.float32).sum(dim=-1)

    def calculate_penalty_collision_avoidance(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        contacts = self._calculate_collision_contacts(
            self.collision_avoidance_links_idx
        )
        if contacts.shape[1] == 0:
            return torch.zeros((self.n_envs,), device=self.device)
        # Weaker penalty: normalized contact ratio.
        return contacts.to(torch.float32).mean(dim=-1)

    def calculate_penalty_collision_dofs_limit(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        del terminated, truncated
        lower_violation = self.state.dofs_position < (
            self.dofs_lower_limit + self.reward_params.collision_dofs_limit_threshold
        )
        upper_violation = self.state.dofs_position > (
            self.dofs_upper_limit - self.reward_params.collision_dofs_limit_threshold
        )
        violation = lower_violation | upper_violation
        return violation.any(dim=-1).to(torch.float32)

    def extract_observation(self) -> TensorDict:
        return TensorDict(
            {
                "base_xyz_velocity": self.state.base_xyz_velocity,
                "base_rpy_velocity": self.state.base_rpy_velocity,
                "dofs_position": self.state.dofs_position - self.dofs_origin,
                "dofs_velocity": self.state.dofs_velocity,
                "projected_gravity": self.state.projected_gravity,
                "gait_phase_sin": self.state.gait_phase.sin(),
                "gait_phase_cos": self.state.gait_phase.cos(),
                "command": self.command,
                "action": self.action_applied,
                "episode_progress": self.episode_steps.to(torch.float32).unsqueeze(-1)
                / float(self.sim_params.max_episode_length),
                "curriculum_level": self.curriculum_level.unsqueeze(-1),
            },
            batch_size=(self.n_envs,),
            device=self.device,
        )

    def calculate_reset_fall_down(self) -> torch.Tensor:
        roll = self.state.base_rotation[:, 0].abs()
        pitch = self.state.base_rotation[:, 1].abs()
        return (roll >= self.reward_params.fall_down_threshold_radians) | (
            pitch >= self.reward_params.fall_down_threshold_radians
        )

    def calculate_reset_timeout(self) -> torch.Tensor:
        if not self.sim_params.shuffle_reset:
            return self.episode_steps >= self.sim_params.max_episode_length

        begin_step = self.sim_params.begin_reset_episode_length
        diff_max = self.sim_params.max_episode_length - begin_step
        diff_step = self.episode_steps - begin_step
        reset_prob = (
            1.0
            / torch.clamp(diff_max - diff_step, min=1)
            * (diff_step >= 0).to(torch.float32)
        )
        return torch.rand((self.n_envs,), device=self.device) < reset_prob

    def calculate_reset_unsafe_xyz_velocity(self) -> torch.Tensor:
        xyz_vel = self.state.base_xyz_velocity.norm(dim=-1)
        return (xyz_vel > 50) | xyz_vel.isnan()

    def calculate_reset_unsafe_rpy_velocity(self) -> torch.Tensor:
        rpy_vel = self.state.base_rpy_velocity.norm(dim=-1)
        return (rpy_vel > 50) | rpy_vel.isnan()

    def calculate_reset_outside_field(self) -> torch.Tensor:
        min_x, max_x, min_y, max_y = self.field_context.get_bounds()
        outside_x = (self.state.base_position[:, 0] < min_x) | (
            self.state.base_position[:, 0] > max_x
        )
        outside_y = (self.state.base_position[:, 1] < min_y) | (
            self.state.base_position[:, 1] > max_y
        )
        return outside_x | outside_y

    def calculate_reset_collision_reset(self) -> torch.Tensor:
        contacts = self._calculate_collision_contacts(self.collision_reset_links_idx)
        if contacts.shape[1] == 0:
            return torch.zeros((self.n_envs,), dtype=torch.bool, device=self.device)
        return contacts.any(dim=-1)

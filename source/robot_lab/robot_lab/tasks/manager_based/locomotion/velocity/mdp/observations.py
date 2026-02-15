# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def action_mask(
    env: ManagerBasedEnv,
    leg_dim: int = 12,
    wheel_dim: int = 4,
    wheel_asset_index: int = 1,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Per-environment action mask for leg+wheel action space."""
    from .utils import get_action_mask

    return get_action_mask(
        env, leg_dim=leg_dim, wheel_dim=wheel_dim, wheel_asset_index=wheel_asset_index, asset_name=asset_name
    )


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def goal_relative_xyz(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the relative target position (xyz) from a pose command."""
    return env.command_manager.get_command(command_name)[:, :3]


def front_elevation_map(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float = 5.0,
    offset: float = 0.5,
) -> torch.Tensor:
    """Return front 180-degree elevation map within max_distance."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    cache_name = f"_front_elevation_ids_{sensor_cfg.name}_{int(max_distance * 1000)}"
    if not hasattr(env, cache_name):
        ray_starts, _ = sensor.cfg.pattern_cfg.func(sensor.cfg.pattern_cfg, env.device)
        ray_xy = ray_starts[:, :2] + torch.tensor(sensor.cfg.offset.pos[:2], device=env.device)
        front_half = ray_xy[:, 0] >= -1.0e-6
        in_range = torch.linalg.norm(ray_xy, dim=1) <= (max_distance + 1.0e-6)
        ray_ids = torch.where(front_half & in_range)[0]
        setattr(env, cache_name, ray_ids)

    ray_ids = getattr(env, cache_name)
    return heights[:, ray_ids]


def history_relative_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    history_duration_s: float = 5.0,
    sample_interval_s: float = 0.5,
) -> torch.Tensor:
    """Return sampled history of past relative pose as [pos_xyz, rpy] sequence."""
    asset: Articulation = env.scene[asset_cfg.name]
    num_samples = max(int(round(history_duration_s / sample_interval_s)), 1)
    sample_interval_steps = max(int(round(sample_interval_s / env.step_dt)), 1)

    pos_attr = f"_pose_history_pos_{asset_cfg.name}_{num_samples}"
    quat_attr = f"_pose_history_quat_{asset_cfg.name}_{num_samples}"
    last_step_attr = f"_pose_history_last_step_{asset_cfg.name}_{num_samples}_{sample_interval_steps}"

    history_pos = getattr(env, pos_attr, None)
    history_quat = getattr(env, quat_attr, None)
    needs_init = (
        history_pos is None
        or history_quat is None
        or history_pos.shape != (env.num_envs, num_samples, 3)
        or history_quat.shape != (env.num_envs, num_samples, 4)
    )
    if needs_init:
        history_pos = asset.data.root_pos_w.unsqueeze(1).repeat(1, num_samples, 1)
        history_quat = asset.data.root_quat_w.unsqueeze(1).repeat(1, num_samples, 1)
        setattr(env, pos_attr, history_pos)
        setattr(env, quat_attr, history_quat)
        setattr(env, last_step_attr, -1)

    current_step = int(getattr(env, "common_step_counter", 0))
    last_step = int(getattr(env, last_step_attr, -1))
    if current_step != last_step:
        episode_length_buf = getattr(env, "episode_length_buf", None)
        if episode_length_buf is not None:
            reset_env_ids = torch.where(episode_length_buf == 0)[0]
            if len(reset_env_ids) > 0:
                history_pos[reset_env_ids] = asset.data.root_pos_w[reset_env_ids].unsqueeze(1).repeat(1, num_samples, 1)
                history_quat[reset_env_ids] = (
                    asset.data.root_quat_w[reset_env_ids].unsqueeze(1).repeat(1, num_samples, 1)
                )

        if current_step % sample_interval_steps == 0:
            history_pos = torch.roll(history_pos, shifts=-1, dims=1)
            history_quat = torch.roll(history_quat, shifts=-1, dims=1)
            history_pos[:, -1] = asset.data.root_pos_w
            history_quat[:, -1] = asset.data.root_quat_w
            setattr(env, pos_attr, history_pos)
            setattr(env, quat_attr, history_quat)

        setattr(env, last_step_attr, current_step)

    current_pos = asset.data.root_pos_w.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 3)
    current_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)
    rel_pos, rel_quat = subtract_frame_transforms(
        current_pos,
        current_quat,
        history_pos.reshape(-1, 3),
        history_quat.reshape(-1, 4),
    )
    rel_pos = rel_pos.view(env.num_envs, num_samples, 3)
    roll, pitch, yaw = euler_xyz_from_quat(rel_quat)
    rel_rpy = torch.stack((roll, pitch, yaw), dim=-1).view(env.num_envs, num_samples, 3)

    return torch.cat((rel_pos, rel_rpy), dim=-1).view(env.num_envs, -1)

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_reached(env: ManagerBasedRLEnv, command_name: str, distance_threshold: float = 0.35) -> torch.Tensor:
    """Terminate when the relative goal distance is within threshold."""
    goal_distance = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    return goal_distance < distance_threshold


def bad_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll: float = 1.0,
    max_pitch: float = 1.0,
) -> torch.Tensor:
    """Terminate when roll/pitch exceeds safety limits."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    return (torch.abs(roll) > max_roll) | (torch.abs(pitch) > max_pitch)


def stuck_without_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_s: float = 3.0,
    move_threshold: float = 0.2,
    progress_threshold: float = 0.1,
    min_goal_distance: float = 0.75,
) -> torch.Tensor:
    """Terminate when robot remains stuck over the configured time window."""
    asset: RigidObject = env.scene[asset_cfg.name]
    current_distance = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    current_step = int(getattr(env, "common_step_counter", 0))
    window_steps = max(int(round(window_s / env.step_dt)), 2)

    pos_hist_attr = f"_stuck_pos_hist_done_{command_name}_{asset_cfg.name}_{window_steps}"
    dist_hist_attr = f"_stuck_dist_hist_done_{command_name}_{asset_cfg.name}_{window_steps}"
    step_attr = f"_stuck_hist_step_done_{command_name}_{asset_cfg.name}_{window_steps}"
    cache_attr = f"_stuck_cache_done_{command_name}_{asset_cfg.name}_{window_steps}"

    pos_hist = getattr(env, pos_hist_attr, None)
    dist_hist = getattr(env, dist_hist_attr, None)
    if (
        pos_hist is None
        or dist_hist is None
        or pos_hist.shape != (env.num_envs, window_steps, 3)
        or dist_hist.shape != (env.num_envs, window_steps)
    ):
        pos_hist = asset.data.root_pos_w.unsqueeze(1).repeat(1, window_steps, 1)
        dist_hist = current_distance.unsqueeze(1).repeat(1, window_steps)
        setattr(env, pos_hist_attr, pos_hist)
        setattr(env, dist_hist_attr, dist_hist)
        setattr(env, step_attr, current_step)
        zero_cache = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, cache_attr, zero_cache)
        return zero_cache

    stuck_cache = getattr(env, cache_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    last_step = int(getattr(env, step_attr, -1))
    if current_step != last_step:
        episode_length_buf = getattr(env, "episode_length_buf", None)
        if episode_length_buf is not None:
            reset_env_ids = torch.where(episode_length_buf == 0)[0]
            if len(reset_env_ids) > 0:
                pos_hist[reset_env_ids] = asset.data.root_pos_w[reset_env_ids].unsqueeze(1).repeat(1, window_steps, 1)
                dist_hist[reset_env_ids] = current_distance[reset_env_ids].unsqueeze(1).repeat(1, window_steps)

        pos_hist = torch.roll(pos_hist, shifts=-1, dims=1)
        dist_hist = torch.roll(dist_hist, shifts=-1, dims=1)
        pos_hist[:, -1] = asset.data.root_pos_w
        dist_hist[:, -1] = current_distance
        setattr(env, pos_hist_attr, pos_hist)
        setattr(env, dist_hist_attr, dist_hist)

        move_distance = torch.linalg.norm(pos_hist[:, -1, :2] - pos_hist[:, 0, :2], dim=1)
        progress = dist_hist[:, 0] - dist_hist[:, -1]
        stuck_cache = (move_distance < move_threshold) & (progress < progress_threshold) & (
            dist_hist[:, -1] > min_goal_distance
        )
        setattr(env, cache_attr, stuck_cache)
        setattr(env, step_attr, current_step)

    return stuck_cache


def unsafe_terrain_ahead(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    max_distance: float = 1.5,
    danger_height: float = 0.45,
    danger_ratio: float = 0.25,
    offset: float = 0.5,
) -> torch.Tensor:
    """Terminate when dangerous terrain fraction ahead exceeds ratio threshold."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    ray_ids_attr = f"_unsafe_terrain_ray_ids_{sensor_cfg.name}_{int(max_distance * 1000)}"
    ray_ids = getattr(env, ray_ids_attr, None)
    if ray_ids is None:
        ray_starts, _ = sensor.cfg.pattern_cfg.func(sensor.cfg.pattern_cfg, env.device)
        ray_xy = ray_starts[:, :2] + torch.tensor(sensor.cfg.offset.pos[:2], device=env.device)
        front_half = ray_xy[:, 0] >= -1.0e-6
        in_range = torch.linalg.norm(ray_xy, dim=1) <= (max_distance + 1.0e-6)
        ray_ids = torch.where(front_half & in_range)[0]
        setattr(env, ray_ids_attr, ray_ids)

    front_heights = heights[:, ray_ids]
    danger_fraction = (torch.abs(front_heights) > danger_height).float().mean(dim=1)
    return danger_fraction > danger_ratio

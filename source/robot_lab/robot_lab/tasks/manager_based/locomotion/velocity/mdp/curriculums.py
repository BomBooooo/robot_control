# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def terrain_levels_pathfinding_success(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    termination_term_name: str = "goal_reached",
    success_rate_up_threshold: float = 0.7,
    success_rate_down_threshold: float = 0.3,
    success_rate_ema_alpha: float = 0.2,
    min_reset_envs: int = 32,
) -> dict[str, float]:
    """Terrain curriculum for pathfinding based on goal-reaching success rate.

    Behavior:
    - Compute success rate on environments being reset using a termination term (default: ``goal_reached``).
    - Maintain an exponential moving average (EMA) of success rate.
    - If EMA is high, successful episodes move to harder terrain.
    - If EMA is low, failed episodes move to easier terrain.
    """

    terrain: TerrainImporter = env.scene.terrain

    if isinstance(env_ids, slice):
        env_ids_tensor = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_tensor = env_ids.to(device=env.device, dtype=torch.long)
    else:
        env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if not hasattr(env, "_pf_success_rate_ema"):
        env._pf_success_rate_ema = 0.0

    # No curriculum update on initial reset before any rollout step.
    if env.common_step_counter == 0 or env_ids_tensor.numel() == 0:
        return {
            "success_rate_batch": 0.0,
            "success_rate_ema": float(env._pf_success_rate_ema),
            "num_resets": float(env_ids_tensor.numel()),
            "mean_terrain_level": float(
                torch.mean(terrain.terrain_levels.float()).item() if terrain.terrain_origins is not None else 0.0
            ),
        }

    # If terrain is not curriculum terrain (e.g., flat plane), skip safely.
    if terrain.terrain_origins is None:
        return {
            "success_rate_batch": 0.0,
            "success_rate_ema": float(env._pf_success_rate_ema),
            "num_resets": float(env_ids_tensor.numel()),
            "mean_terrain_level": 0.0,
        }

    if termination_term_name not in env.termination_manager.active_terms:
        return {
            "success_rate_batch": 0.0,
            "success_rate_ema": float(env._pf_success_rate_ema),
            "num_resets": float(env_ids_tensor.numel()),
            "mean_terrain_level": float(torch.mean(terrain.terrain_levels.float()).item()),
        }

    goal_reached = env.termination_manager.get_term(termination_term_name)[env_ids_tensor].bool()
    success_rate_batch = goal_reached.float().mean().item()
    env._pf_success_rate_ema = (
        (1.0 - success_rate_ema_alpha) * float(env._pf_success_rate_ema) + success_rate_ema_alpha * success_rate_batch
    )

    move_up = torch.zeros_like(goal_reached)
    move_down = torch.zeros_like(goal_reached)

    if env_ids_tensor.numel() >= min_reset_envs:
        if env._pf_success_rate_ema >= success_rate_up_threshold:
            move_up = goal_reached
        elif env._pf_success_rate_ema <= success_rate_down_threshold:
            move_down = ~goal_reached

    terrain.update_env_origins(env_ids_tensor, move_up, move_down)

    return {
        "success_rate_batch": float(success_rate_batch),
        "success_rate_ema": float(env._pf_success_rate_ema),
        "num_resets": float(env_ids_tensor.numel()),
        "mean_terrain_level": float(torch.mean(terrain.terrain_levels.float()).item()),
    }

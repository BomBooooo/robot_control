# Mixed Go2 / Go2W Task IO Summary

This document summarizes the current task inputs/outputs derived from the config.
If observation terms or sensors change, the dimensions here must be updated.

## Actions (Policy Output)
- Action dim: 16
- Layout:
  - 0..11: 12 leg joints (FR/FL/RR/RL hip, thigh, calf)
  - 12..15: 4 wheel joints (FR/FL/RR/RL foot joints)
- Go2 (no wheels): wheel actions are masked to zero (per-env mask).

## Policy Observations (Rough / Flat)
Terms (order preserved, concatenated):
- base_ang_vel: 3
- projected_gravity: 3
- velocity_commands (base_velocity): 3
- joint_pos: 16
- joint_vel: 16
- actions (last action): 16
- action_mask: 16

Total dim: 73

Notes:
- base_lin_vel is disabled for policy.
- height_scan is disabled for policy in mixed task.

## Critic Observations (Rough)
Terms (order preserved, concatenated):
- base_lin_vel: 3
- base_ang_vel: 3
- projected_gravity: 3
- velocity_commands (base_velocity): 3
- joint_pos: 16
- joint_vel: 16
- actions (last action): 16
- action_mask: 16
- height_scan: 160  (grid 1.6 x 1.0 @ 0.1 m resolution)

Total dim: 236

## Critic Observations (Flat)
Same as rough, but height_scan is disabled:

Total dim: 76

## References
- Task config: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py`
- Base velocity env config: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

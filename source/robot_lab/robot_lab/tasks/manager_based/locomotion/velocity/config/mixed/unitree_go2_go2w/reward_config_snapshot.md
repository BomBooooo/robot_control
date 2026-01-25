# Reward Configuration Snapshot (Mixed Go2/Go2W)

This snapshot captures the current reward configuration and mixed-robot masking logic for the mixed Go2/Go2W task.

## 1) Mixed Task Reward Overrides
Source: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py`

### 1.1 Base Pose / Velocity Tracking (masked per asset index)
- `track_lin_vel_xy_exp_go2`: `track_lin_vel_xy_exp_masked`, weight **3.0**, params: `command_name=base_velocity`, `std=sqrt(0.25)`, `asset_index=0`
- `track_lin_vel_xy_exp_go2w`: `track_lin_vel_xy_exp_masked`, weight **3.0**, params: `command_name=base_velocity`, `std=sqrt(0.25)`, `asset_index=1`
- `track_ang_vel_z_exp_go2`: `track_ang_vel_z_exp_masked`, weight **1.5**, params: `command_name=base_velocity`, `std=sqrt(0.25)`, `asset_index=0`
- `track_ang_vel_z_exp_go2w`: `track_ang_vel_z_exp_masked`, weight **1.5**, params: `command_name=base_velocity`, `std=sqrt(0.25)`, `asset_index=1`

- `lin_vel_z_l2_go2`: `lin_vel_z_l2_masked`, weight **-2.0**, params: `asset_cfg=robot`, `asset_index=0`
- `lin_vel_z_l2_go2w`: `lin_vel_z_l2_masked`, weight **-2.0**, params: `asset_cfg=robot`, `asset_index=1`
- `ang_vel_xy_l2_go2`: `ang_vel_xy_l2_masked`, weight **-0.05**, params: `asset_cfg=robot`, `asset_index=0`
- `ang_vel_xy_l2_go2w`: `ang_vel_xy_l2_masked`, weight **-0.05**, params: `asset_cfg=robot`, `asset_index=1`

- `base_height_l2_go2`: `base_height_l2_masked`, weight **0**, target_height **0.33**, `asset_index=0`
- `base_height_l2_go2w`: `base_height_l2_masked`, weight **0**, target_height **0.40**, `asset_index=1`

### 1.2 Wheel-specific (Go2W only)
- `joint_torques_wheel_l2`: `joint_torques_l2_masked`, weight **0**, `asset_index=1`
- `joint_vel_wheel_l2`: `joint_vel_l2_masked`, weight **0**, `asset_index=1`
- `joint_acc_wheel_l2`: `joint_acc_l2_masked`, weight **-2.5e-9**, `asset_index=1`
- `wheel_vel_penalty`: `wheel_vel_penalty_masked`, weight **0**, `asset_index=1`

### 1.3 Leg-specific (Go2 only)
- `feet_air_time_go2`: `feet_air_time_masked`, weight **0.1**, `asset_index=0`
- `feet_air_time_variance_go2`: `feet_air_time_variance_masked`, weight **-1.0**, `asset_index=0`
- `feet_slide_go2`: `feet_slide_masked`, weight **-0.1**, `asset_index=0`
- `feet_height_body_go2`: `feet_height_body_masked`, weight **-5.0**, `asset_index=0`
- `feet_gait_go2`: `MaskedGaitReward`, weight **0.5**, `asset_index=0`

### 1.4 Shared (unmasked or identical per-robot)
Weights are the same for both robots, so the base (unmasked) terms remain active:
- `joint_torques_l2` (legs) **-2.5e-5**
- `joint_vel_l2` (legs) **0**
- `joint_acc_l2` (legs) **-2.5e-7**
- `joint_pos_limits` **-5.0**
- `joint_power` **-2e-5**
- `stand_still` **-2.0**
- `joint_pos_penalty` **-1.0**
- `joint_mirror` **-0.05**
- `action_rate_l2` **-0.01**
- `undesired_contacts` **-1.0**
- `contact_forces` **-1.5e-4**
- `feet_contact_without_cmd` **0.1**
- `upward` **1.0**

## 2) Masked Reward Helpers (Logic)
Source: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`

Masked helpers:
- `track_lin_vel_xy_exp_masked`
- `track_ang_vel_z_exp_masked`
- `lin_vel_z_l2_masked`
- `ang_vel_xy_l2_masked`
- `base_height_l2_masked`
- `joint_torques_l2_masked`
- `joint_vel_l2_masked`
- `joint_acc_l2_masked`
- `wheel_vel_penalty_masked`
- `feet_air_time_masked`
- `feet_air_time_variance_masked`
- `feet_slide_masked`
- `feet_height_body_masked`
- `action_rate_l2_masked`
- `MaskedGaitReward`

Masking uses `get_robot_asset_indices()` to select by `asset_index` (Go2=0, Go2W=1).

## 3) Base Reward Template (Inherited)
Source: `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

All reward terms are defined in `RewardsCfg`; mixed task overrides weights/params in its rough config. The mixed task also disables zero-weight rewards via `disable_zero_weight_rewards()`.

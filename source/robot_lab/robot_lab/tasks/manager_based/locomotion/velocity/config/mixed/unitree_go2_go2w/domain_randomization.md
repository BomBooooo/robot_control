# Domain Randomization Summary (Go2/Go2W Mixed Task)

| Category | Trigger | Item | Parameters / Range | Source |
| --- | --- | --- | --- | --- |
| Physics | startup | Rigid body material | static_friction_range=(0.3, 1.0); dynamic_friction_range=(0.3, 0.8); restitution_range=(0.0, 0.5); num_buckets=64 | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:258 |
| Physics | startup | Base mass randomization | mass_distribution_params=(-1.0, 3.0); operation="add"; recompute_inertia=True | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:270; source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py:150 |
| Physics | startup | Other body mass randomization | mass_distribution_params=(0.7, 1.3); operation="scale"; recompute_inertia=True | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:281; source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py:153 |
| Physics | startup | COM position randomization | com_range={x=(-0.05, 0.05), y=(-0.05, 0.05), z=(-0.05, 0.05)} | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:301; source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py:156 |
| Dynamics | reset | External force/torque | force_range=(-10.0, 10.0); torque_range=(-10.0, 10.0) | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:313; source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py:158 |
| Dynamics | reset | Joint reset (scale) | position_range=(1.0, 1.0); velocity_range=(0.0, 0.0) | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:325 |
| Dynamics | reset | Actuator gains (scale) | stiffness_distribution_params=(0.5, 2.0); damping_distribution_params=(0.5, 2.0); distribution="uniform" | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:335 |
| State | reset | Base pose/velocity | pose_range: x/y=(-0.5, 0.5), z=(0.0, 0.2), roll/pitch/yaw=(-3.14, 3.14); velocity_range: x/y/z/roll/pitch/yaw=(-0.5, 0.5) | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/rough_env_cfg.py:136 |
| Perturbation | interval | Random pushes | interval_range_s=(10.0, 15.0); velocity_range: x/y=(-0.5, 0.5) | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:366 |
| Asset mix | startup | Go2 vs Go2W assignment | asset_ratios=[1.0, 1.0]; randomize=True (1:1 per batch; shuffles env assignment) | source/robot_lab/robot_lab/assets/unitree.py (UNITREE_GO2_GO2W_CFG) |
| Terrain | startup | Rough terrain generator | terrain_type="generator"; terrain_generator=ROUGH_TERRAINS_CFG (randomized terrain) | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:38 |
| Terrain | startup | Flat terrain (flat config only) | terrain_type="plane"; terrain_generator=None | source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/mixed/unitree_go2_go2w/flat_env_cfg.py:17 |

Notes:
- Items inherited from `velocity_env_cfg.py` apply to the mixed task unless overridden in the mixed config.
- If `num_envs` is odd, the extra environment is assigned to Go2 (asset index 0).

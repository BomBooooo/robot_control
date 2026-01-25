# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_GO2_GO2W_CFG  # isort: skip


@configclass
class UnitreeGo2Go2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.MaskedJointVelocityActionCfg(
        asset_name="robot",
        joint_names=[""],
        scale=5.0,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
        wheel_asset_index=1,
    )


@configclass
class UnitreeGo2Go2WRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # Base pose / velocity tracking (masked by asset index)
    track_lin_vel_xy_exp_go2 = RewTerm(
        func=mdp.track_lin_vel_xy_exp_masked,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_index": 0},
    )
    track_lin_vel_xy_exp_go2w = RewTerm(
        func=mdp.track_lin_vel_xy_exp_masked,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_index": 1},
    )
    track_ang_vel_z_exp_go2 = RewTerm(
        func=mdp.track_ang_vel_z_exp_masked,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_index": 0},
    )
    track_ang_vel_z_exp_go2w = RewTerm(
        func=mdp.track_ang_vel_z_exp_masked,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_index": 1},
    )
    lin_vel_z_l2_go2 = RewTerm(
        func=mdp.lin_vel_z_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "asset_index": 0},
    )
    lin_vel_z_l2_go2w = RewTerm(
        func=mdp.lin_vel_z_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "asset_index": 1},
    )
    ang_vel_xy_l2_go2 = RewTerm(
        func=mdp.ang_vel_xy_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "asset_index": 0},
    )
    ang_vel_xy_l2_go2w = RewTerm(
        func=mdp.ang_vel_xy_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "asset_index": 1},
    )
    base_height_l2_go2 = RewTerm(
        func=mdp.base_height_l2_masked,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.33,
            "asset_index": 0,
        },
    )
    base_height_l2_go2w = RewTerm(
        func=mdp.base_height_l2_masked,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.40,
            "asset_index": 1,
        },
    )

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=""), "asset_index": 1},
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=""), "asset_index": 1},
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2_masked,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=""), "asset_index": 1},
    )

    # Leg-related (Go2-specific) rewards
    feet_air_time_go2 = RewTerm(
        func=mdp.feet_air_time_masked,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_index": 0,
        },
    )
    feet_air_time_variance_go2 = RewTerm(
        func=mdp.feet_air_time_variance_masked,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "asset_index": 0},
    )
    feet_slide_go2 = RewTerm(
        func=mdp.feet_slide_masked,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "asset_index": 0,
        },
    )
    feet_height_body_go2 = RewTerm(
        func=mdp.feet_height_body_masked,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.2,
            "command_name": "base_velocity",
            "asset_index": 0,
        },
    )
    feet_gait_go2 = RewTerm(
        func=mdp.MaskedGaitReward,
        weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_index": 0,
        },
    )


@configclass
class UnitreeGo2Go2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: UnitreeGo2Go2WActionsCfg = UnitreeGo2Go2WActionsCfg()
    rewards: UnitreeGo2Go2WRewardsCfg = UnitreeGo2Go2WRewardsCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"

    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # heterogeneous assets (go2/go2w)
        self.scene.replicate_physics = False

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.action_mask = ObsTerm(
            func=mdp.action_mask,
            params={
                "leg_dim": len(self.leg_joint_names),
                "wheel_dim": len(self.wheel_joint_names),
                "wheel_asset_index": 1,
            },
            scale=1.0,
        )
        self.observations.critic.action_mask = ObsTerm(
            func=mdp.action_mask,
            params={
                "leg_dim": len(self.leg_joint_names),
                "wheel_dim": len(self.wheel_joint_names),
                "wheel_asset_index": 1,
            },
            scale=1.0,
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.lin_vel_z_l2_go2.weight = -2.0
        self.rewards.lin_vel_z_l2_go2w.weight = -2.0
        self.rewards.ang_vel_xy_l2_go2.weight = -0.05
        self.rewards.ang_vel_xy_l2_go2w.weight = -0.05
        self.rewards.base_height_l2_go2.weight = 0
        self.rewards.base_height_l2_go2w.weight = 0
        self.rewards.base_height_l2_go2.params["target_height"] = 0.33
        self.rewards.base_height_l2_go2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2_go2.params["sensor_cfg"].name = "height_scanner_base"
        self.rewards.base_height_l2_go2w.params["target_height"] = 0.40
        self.rewards.base_height_l2_go2w.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2_go2w.params["sensor_cfg"].name = "height_scanner_base"

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_vel_penalty.func = mdp.wheel_vel_penalty_masked
        self.rewards.wheel_vel_penalty.params["asset_index"] = 1
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.rewards.track_lin_vel_xy_exp_go2.weight = 3.0
        self.rewards.track_lin_vel_xy_exp_go2w.weight = 3.0
        self.rewards.track_ang_vel_z_exp_go2.weight = 1.5
        self.rewards.track_ang_vel_z_exp_go2w.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_go2.weight = 0.1
        self.rewards.feet_air_time_go2.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance_go2.weight = -1.0
        self.rewards.feet_air_time_variance_go2.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_go2.weight = -0.1
        self.rewards.feet_slide_go2.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_go2.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body_go2.weight = -5.0
        self.rewards.feet_height_body_go2.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.feet_gait_go2.weight = 0.5
        self.rewards.feet_gait_go2.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2Go2WRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

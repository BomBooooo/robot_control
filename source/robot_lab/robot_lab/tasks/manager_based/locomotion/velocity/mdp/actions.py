# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import actions_cfg
from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .utils import get_wheel_action_mask


class MaskedJointVelocityAction(JointVelocityAction):
    """Joint velocity action term with per-environment masking."""

    cfg: "MaskedJointVelocityActionCfg"

    def process_actions(self, actions: torch.Tensor):
        wheel_mask = get_wheel_action_mask(
            self._env,
            wheel_dim=actions.shape[1],
            wheel_asset_index=self.cfg.wheel_asset_index,
            asset_name=self.cfg.asset_name,
        )
        super().process_actions(actions * wheel_mask)


@configclass
class MaskedJointVelocityActionCfg(actions_cfg.JointVelocityActionCfg):
    """Configuration for masked joint velocity action."""

    class_type: type[ActionTerm] = MaskedJointVelocityAction

    wheel_asset_index: int = 1
    """Asset index treated as wheel-equipped (mask=1)."""

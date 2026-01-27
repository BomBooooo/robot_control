# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

# Register Gym environments.

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-Go2-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2Go2WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPORunnerCfg",
        "rsl_rl_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPORNNRunnerCfg",
        "rsl_rl_gru_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPOGRURunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPOLSTMRunnerCfg",
        "rsl_rl_mamba_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPOMambaRunnerCfg",
        "rsl_rl_xlstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPOXLSTMRunnerCfg",
        "rsl_rl_rwkv_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPORWKVRunnerCfg"
        ),
        "rsl_rl_transformer_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WFlatPPOTransformerRunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2Go2WFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2Go2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPORunnerCfg",
        "rsl_rl_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPORNNRunnerCfg",
        "rsl_rl_gru_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPOGRURunnerCfg",
        "rsl_rl_lstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPOLSTMRunnerCfg",
        "rsl_rl_mamba_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPOMambaRunnerCfg",
        "rsl_rl_xlstm_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPOXLSTMRunnerCfg",
        "rsl_rl_rwkv_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPORWKVRunnerCfg",
        "rsl_rl_transformer_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2Go2WRoughPPOTransformerRunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2Go2WRoughTrainerCfg",
    },
)

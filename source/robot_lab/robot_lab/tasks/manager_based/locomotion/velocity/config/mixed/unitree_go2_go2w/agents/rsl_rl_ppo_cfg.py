# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoActorCriticTransformerCfg(RslRlPpoActorCriticRecurrentCfg):
    transformer_nhead: int | None = 4
    transformer_context_len: int = 32
    transformer_dropout: float = 0.0


@configclass
class RslRlPpoActorCriticRWKVCfg(RslRlPpoActorCriticRecurrentCfg):
    rwkv_head_size: int = 32
    rwkv_time_mix_extra_dim: int = 32
    rwkv_time_decay_extra_dim: int = 64
    rwkv_ffn_expand: int = 3
    rwkv_bias: bool = False
    rwkv_ln_eps: float = 1e-5
    rwkv_chunk_size: int = 256
    xlstm_head_num: int | None = None
    xlstm_head_dim: int | None = None


@configclass
class RslRlPpoActorCriticXLSTMCfg(RslRlPpoActorCriticRecurrentCfg):
    xlstm_head_num: int | None = None
    xlstm_head_dim: int | None = None


@configclass
class UnitreeGo2Go2WRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_go2w_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2Go2WFlatPPORunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat"


@configclass
class UnitreeGo2Go2WRoughPPORNNRunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_rnn"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="rnn",
        rnn_hidden_dim=[640],
        rnn_num_layers=1,
    )


@configclass
class UnitreeGo2Go2WRoughPPOGRURunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_gru"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=[416],
        rnn_num_layers=1,
    )


@configclass
class UnitreeGo2Go2WRoughPPOLSTMRunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_lstm"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=[384],
        rnn_num_layers=1,
    )


@configclass
class UnitreeGo2Go2WRoughPPOMambaRunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_mamba"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="mamba",
        rnn_hidden_dim=[384],
        rnn_num_layers=1,
    )


@configclass
class UnitreeGo2Go2WRoughPPOXLSTMRunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_xlstm"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="xlstm",
        rnn_hidden_dim=[384],
        rnn_num_layers=1,
    )


@configclass
class UnitreeGo2Go2WRoughPPORWKVRunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_rwkv"
    policy = RslRlPpoActorCriticRWKVCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="rwkv",
        rnn_hidden_dim=[384],
        rnn_num_layers=1,
        rwkv_head_size=32,
        rwkv_time_mix_extra_dim=32,
        rwkv_time_decay_extra_dim=64,
        rwkv_ffn_expand=3,
        rwkv_bias=False,
        rwkv_ln_eps=1e-5,
        rwkv_chunk_size=256,
    )


@configclass
class UnitreeGo2Go2WRoughPPOTransformerRunnerCfg(UnitreeGo2Go2WRoughPPORunnerCfg):
    experiment_name = "unitree_go2_go2w_rough_transformer"
    policy = RslRlPpoActorCriticTransformerCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="transformer",
        rnn_hidden_dim=[160],
        rnn_num_layers=1,
        transformer_nhead=4,
        transformer_context_len=32,
        transformer_dropout=0.1,
    )


@configclass
class UnitreeGo2Go2WFlatPPORNNRunnerCfg(UnitreeGo2Go2WRoughPPORNNRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_rnn"


@configclass
class UnitreeGo2Go2WFlatPPOGRURunnerCfg(UnitreeGo2Go2WRoughPPOGRURunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_gru"


@configclass
class UnitreeGo2Go2WFlatPPOLSTMRunnerCfg(UnitreeGo2Go2WRoughPPOLSTMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_lstm"


@configclass
class UnitreeGo2Go2WFlatPPOMambaRunnerCfg(UnitreeGo2Go2WRoughPPOMambaRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_mamba"


@configclass
class UnitreeGo2Go2WFlatPPOXLSTMRunnerCfg(UnitreeGo2Go2WRoughPPOXLSTMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_xlstm"


@configclass
class UnitreeGo2Go2WFlatPPORWKVRunnerCfg(UnitreeGo2Go2WRoughPPORWKVRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_rwkv"


@configclass
class UnitreeGo2Go2WFlatPPOTransformerRunnerCfg(UnitreeGo2Go2WRoughPPOTransformerRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.experiment_name = "unitree_go2_go2w_flat_transformer"

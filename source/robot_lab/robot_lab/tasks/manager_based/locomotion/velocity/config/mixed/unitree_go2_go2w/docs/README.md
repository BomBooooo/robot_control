# Mixed Go2 / Go2W Locomotion Task

This folder defines a mixed training task where Go2 and Go2W share a single 16-DOF action space.
- Action layout: first 12 dims = legs, last 4 dims = wheels.
- For Go2 (no wheels), wheel actions are masked to zero.
- A 16-dim action mask is appended to both policy and critic observations.

## Task IDs
- Rough: `RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0`
- Flat: `RobotLab-Isaac-Velocity-Flat-Unitree-Go2-Go2W-v0`

## Training (RSL-RL)
```bash
# Rough
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0 --headless

# Flat
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-Go2W-v0 --headless
```

## RSL-RL Model Variants
Use `--agent` to switch policy backbones while keeping the same task setup.

Available agents:
- `rsl_rl_cfg_entry_point` (baseline MLP, default)
- `rsl_rl_rnn_cfg_entry_point` (vanilla RNN)
- `rsl_rl_gru_cfg_entry_point` (GRU)
- `rsl_rl_lstm_cfg_entry_point` (LSTM)
- `rsl_rl_mamba_cfg_entry_point` (Mamba)
- `rsl_rl_xlstm_cfg_entry_point` (xLSTM)
- `rsl_rl_rwkv_cfg_entry_point` (RWKV v6, pure PyTorch)
- `rsl_rl_transformer_cfg_entry_point` (Transformer)

Example (rough + GRU):
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0 \
  --agent rsl_rl_gru_cfg_entry_point --headless
```

Example (rough + Transformer with custom params):
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0 \
  --agent rsl_rl_transformer_cfg_entry_point \
  --agent_cfg.policy.transformer_nhead=4 \
  --agent_cfg.policy.transformer_context_len=64 \
  --agent_cfg.policy.transformer_dropout=0.1 \
  --headless
```

Optional dependencies for advanced models:
```bash
pip install mamba-ssm[causal-conv1d]
pip install git+https://github.com/myscience/x-lstm
```

Recurrent layer sizing:
- `rnn_hidden_dim` now supports list form (e.g., `[256, 192, 128]`).
- The list length must match `rnn_num_layers`.
- Variable per-layer dims are supported for RNN/GRU/LSTM. Mamba/xLSTM/RWKV require uniform dims.
- Transformer options (only when `rnn_type="transformer"`):
  - `transformer_nhead` (default 4, must divide `rnn_hidden_dim`).
  - `transformer_context_len` (default 32).
  - `transformer_dropout` (default 0.0).
- RWKV v6 Torch options (only when `rnn_type="rwkv"`):
  - `rwkv_head_size` (default 32; must divide `rnn_hidden_dim`, otherwise falls back to a single head).
  - `rwkv_time_mix_extra_dim` (default 32).
  - `rwkv_time_decay_extra_dim` (default 64).
  - `rwkv_ffn_expand` (default 3).
  - `rwkv_bias` (default false).
  - `rwkv_ln_eps` (default 1e-5).
  - `rwkv_chunk_size` (default 512; PPO update batch chunking to reduce peak VRAM).

## Play (RSL-RL)
```bash
# Rough
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0

# Flat
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-Go2W-v0
```

## Training (CusRL)
```bash
# Rough
python scripts/reinforcement_learning/cusrl/train.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0 --headless

# Flat
python scripts/reinforcement_learning/cusrl/train.py --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-Go2W-v0 --headless
```

## Play (CusRL)
```bash
# Rough
python scripts/reinforcement_learning/cusrl/play.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0

# Flat
python scripts/reinforcement_learning/cusrl/play.py --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-Go2W-v0
```

## Debug Helpers
```bash
# Zero-action agent
python scripts/tools/zero_agent.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0

# Random-action agent
python scripts/tools/random_agent.py --task=RobotLab-Isaac-Velocity-Rough-Unitree-Go2-Go2W-v0
```

## Notes
- Use `--num_envs` to override the default number of environments.
- Use `--video --video_length 200` to record training videos (requires `ffmpeg`).
- Add `--keyboard` at the end of a play command to control a single robot with the keyboard.

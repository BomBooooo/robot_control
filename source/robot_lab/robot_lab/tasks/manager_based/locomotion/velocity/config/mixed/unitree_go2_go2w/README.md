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

## Evaluate you model

## Installation

Follow the official [IsaacLab 2.2 installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to set up **IsaacLab 2.2** (this repository) with **Isaac Sim 5.0**.

## Run evaluation
```bash
cd IsaacLab

./isaaclab.sh -p ./scripts/reinforcement_learning/rl_games/evaluate.py --task Isaac-Factory-Lighter-Direct-v0 --num_envs 16 --headless --log_dir saved_results --video
```
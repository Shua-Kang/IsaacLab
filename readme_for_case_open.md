## Installation

Follow the official [IsaacLab 2.2 installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to set up **IsaacLab 2.2** (this repository) with **Isaac Sim 5.0**.

## Run Case Open task with Depth Tactile
```bash
cd IsaacLab
# Modify this command if you run on Ubuntu.
./isaaclab.bat -p .\scripts\environments\teleoperation\teleop_se3_agent.py --enable_cameras --task Isaac-Factory-Lighter-Direct-v0 --num_envs 1 --teleop_device keyboard```
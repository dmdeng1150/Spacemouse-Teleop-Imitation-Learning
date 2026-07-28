# Spacemouse-Teleop-Imitation-Learning
A virtual environment for testing imitation learning using demonstrations from a spacemouse controlled panda arm. 

## Requirements and Installing Dependencies
For optimal gymnasium and imitation compatibility, use Python versions 3.10 or 3.11 (virtual environment recommended). 
Using pip, install:
- gymnasium >=0.29.0, <1.0.0 (0.29.1 confirmed works)
- imitation >=1.0.0 (1.0.1 confirmed works)
- numpy >=1.26.0, <2.0.0 (1.26.4 confirmed works)
- fastapi >=0.95.0
- uvicorn >=0.20.0
- pyspacemouse >=1.1.2
- easyhid >=0.0.9
- stable_baselines3 2.2.1
- gymnasium_robotics 1.2.2
NOTE: To resolve gymnasium-robotics dependency issues, try installation of all packages with the following single-line command:
pip install "gymnasium<0.30,>=0.29.0" "gymnasium-robotics==1.2.4" "imitation==1.0.1" "stable-baselines3==2.2.1" "numpy<2.0.0"

Finally, install panda_mujoco_gym 0.1.0. To do so, clone repo https://github.com/zichunxx/panda_mujoco_gym in parent directory. Then copy setup.py and pyproject.toml files (proivded in this repo) to root panda_mujoco_gym directory. Run pip install -e . in this directory to complete panda_mujoco_gym installation)

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
```
pip install "gymnasium<0.30,>=0.29.0" "gymnasium-robotics==1.2.4" "imitation==1.0.1" "stable-baselines3==2.2.1" "numpy<2.0.0"
```

Finally, you will need to install panda_mujoco_gym 0.1.0. We created a [fork](https://github.com/alberteks/panda_mujoco_gym) of the original repo*. To use it, clone our repo with submodules:
```
git clone --recursive https://github.com/dmdeng1150/Spacemouse-Teleop-Imitation-Learning.git
```
If you already cloned without --recursive, use
```
git submodule update --init --recursive
```

Then, install our fork of panda_mujoco_gym via
```
pip install -e ./panda_mujoco_gym
```
Install remaining dependencies, using requirements.txt if needed (should already be listed in dependencies list above).

*Note: We forked the original repo since it recalculated the mocap target from the current position every step, causing the Franka arm to return back to neutral when the SpaceMouse input was released/SpaceMouse action was [0,0,0]. Now, the current position is instead immediately held on release.

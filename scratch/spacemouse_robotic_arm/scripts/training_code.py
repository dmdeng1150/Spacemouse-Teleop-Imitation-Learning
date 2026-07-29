import gymnasium as gym
import numpy as np
import pickle
import os
import time
import sys
import pyspacemouse  # Core SpaceMouse driver wrapper
import panda_mujoco_gym 
import threading
import time

class SpaceMouseThread:
    """thread running in background for spacemouse input readings"""
    def __init__(self, device):
        self.device = device
        self.latest_state = None
        self.running = True
        # background (daemon) thread automatically terminates when main script exits
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        while self.running:
            try:
                state = self.device.read()
                if state is not None:
                    self.latest_state = state
            except Exception:
                pass

    def get_state(self):
        return self.latest_state

    def stop(self):
        self.running = False

class FlattenGoalEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        low = np.concatenate([spaces['observation'].low, spaces['achieved_goal'].low, spaces['desired_goal'].low])
        high = np.concatenate([spaces['observation'].high, spaces['achieved_goal'].high, spaces['desired_goal'].high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])

def smooth_deadzone(x, deadzone=0.08):
    if abs(x) < deadzone:
        return 0.0
    return np.sign(x) * (abs(x) - deadzone) / (1.0 - deadzone)

def run_operator_session(env_id="FrankaPickAndPlaceSparse-v0", total_episodes=5, output_file="operator_data.pkl"):
    CONTROL_HZ = 100
    CONTROL_DT = 1.0 / CONTROL_HZ
    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                dataset = pickle.load(f)
            print(f"Loaded {len(dataset)} existing episodes.")
        except Exception:
            dataset = []
    else:
        dataset = []

    initial_count = len(dataset)

    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    env = FlattenGoalEnv(raw_env)

    CTRL_EMA_BETA = 0.05
    _ctrl_target = None

    _franka_env = raw_env.unwrapped
    _original_set_action = _franka_env._set_action

    def _set_action_soft_hold(action):
        nonlocal _ctrl_target
        _original_set_action(action)
        current_q = np.array([
            _franka_env._utils.get_joint_qpos(_franka_env.model, _franka_env.data, name) 
            for name in _franka_env.arm_joint_names
        ]).flatten()
        if _ctrl_target is None:
            _ctrl_target = current_q.copy()
        else:
            _ctrl_target = (1 - CTRL_EMA_BETA) * _ctrl_target + CTRL_EMA_BETA * current_q
            _franka_env.data.ctrl[0:7] = _ctrl_target

    _franka_env._set_action = _set_action_soft_hold

    SENSITIVITY = 0.25 # unitless sensitivity factor
    MAX_ACCEL = 6.0 # in action-units/s^2
    GLITCH_FILTER_ALPHA = 0.55 # flattens any short glitches from spring effect after letting go of mouse
    
    with pyspacemouse.open() as device:
        sm_thread = SpaceMouseThread(device)
        
        for ep in range(total_episodes):
            obs, info = env.reset()
            current_action_xyz = np.zeros(3, dtype=np.float32)
            done = False
            aborted = False
            terminated = False
            truncated = False

            
            ep_obs = [obs]
            ep_acts = []
            gripper_val = 1.0
            step_count = 0
            
            current_ep_num = initial_count + ep + 1
            print(f"\n--- Recording Episode {current_ep_num} ---")

            filtered_raw = {"x": 0.0, "y": 0.0, "z": 0.0} # filter to avoid snapback

            next_tick = time.perf_counter()

            while not done:
                state = sm_thread.get_state()
                if state is None:
                    time.sleep(0.002)
                    continue

                # Gripper control buttons
                left, right = state.buttons
                if left and right:
                    print(f"Episode {ep + 1} flagged complete by operator.")
                    aborted = True
                    break
                elif left:
                    gripper_val = -1.0
                elif right:
                    gripper_val = 1.0

                filtered_raw["x"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["x"] + GLITCH_FILTER_ALPHA * state.x
                filtered_raw["y"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["y"] + GLITCH_FILTER_ALPHA * state.y
                filtered_raw["z"] = (1 - GLITCH_FILTER_ALPHA) * filtered_raw["z"] + GLITCH_FILTER_ALPHA * state.z

                # desired action read from SpaceMouse
                desired_action_xyz = np.array([
                    smooth_deadzone(filtered_raw["x"], deadzone=0.12),
                    smooth_deadzone(filtered_raw["y"], deadzone=0.12),
                    smooth_deadzone(filtered_raw["z"], deadzone=0.12),
                ], dtype=np.float32)

                desired_action_xyz *= SENSITIVITY

                # slew-rate limiter (limits acceleration; namely, how fast the action can change)
                action_change = desired_action_xyz - current_action_xyz

                max_change = MAX_ACCEL * CONTROL_DT

                action_change = np.clip(
                    action_change,
                    -max_change,
                    max_change,
                )

                current_action_xyz += action_change

                # Keep within action limits
                xyz_action = np.clip(current_action_xyz, -1.0, 1.0)

                action = np.array([
                    xyz_action[1],
                    xyz_action[0],
                    xyz_action[2],
                    gripper_val,
                ], dtype=np.float32)

                # environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1

                ep_obs.append(next_obs)
                ep_acts.append(action)

                next_tick += CONTROL_DT

                sleep = next_tick - time.perf_counter()

                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
                
            if aborted:
                print(f"Trial {current_ep_num} DISCARDED (Cancelled by operator).")
            elif truncated:
                print(f"Trial {current_ep_num} DISCARDED (Reached max episode steps).")
            elif terminated:
                dataset.append({
                "obs": np.array(ep_obs, dtype=np.float32),
                "acts": np.array(ep_acts, dtype=np.float32),
                "terminal": True
                })
                print(f"Saved Episode {current_ep_num} with {len(ep_acts)} steps!")
            else:
                print(f"Trial {current_ep_num} DISCARDED (Ended without success).")
    env.close()
    
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    run_operator_session(env_id="FrankaPickAndPlaceSparse-v0", total_episodes=5)
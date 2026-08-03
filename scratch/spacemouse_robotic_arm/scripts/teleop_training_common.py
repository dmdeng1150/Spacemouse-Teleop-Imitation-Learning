# shared file for all input methods used for training. 
# determines which task is being performed, gives input method interface, and implements operator method
import os
import pickle
import time
import numpy as np
from scipy.signal import savgol_filter
import gymnasium as gym
import panda_mujoco_gym

from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper

CONTROL_HZ = 100 # run polling at 100 hz
CONTROL_DT = 1.0 / CONTROL_HZ

TASK_CONFIG = {
    "push": {
        "env_id": "FrankaPushSparse-v0",
        "act_dim": 3,
    },
    "pick_and_place": {
        "env_id": "FrankaPickAndPlaceSparse-v0",
        "act_dim": 4,
    },
}

def prompt_for_task() -> str:
    # get user input to determine which task we are teleoping for
    raw = input("Select task -- Push (1) or Pick-and-Place (2): ").strip()
    return {"1": "push", "2": "pick_and_place"}.get(raw)

# interface class for both spacemouse and keyboard input methods
class InputSource:
    # methods meant to be implemented by classes using this interface (input sources)
    def get_action(self, act_dim: int) -> np.ndarray: # return unclipped action vector of len act_dim
        raise NotImplementedError
    
    def should_abort(self) -> bool:
        raise NotImplementedError

    def reset_state(self) -> None:
        raise NotImplementedError

    def on_env_ready(self, env) -> None: # this method is optional, so we just pass if it isn't implemented in class. called once render window exists
        pass

    def close(self) -> None:
        raise NotImplementedError

def _filter_and_save(existing_dataset, new_dataset, output_file):
    # filtering and smoothing data
    print("\n--- Applying dataset filters (smoothing & cleaning) ---")
    filtered_new_dataset = []

    for traj in new_dataset:
        raw_obs = traj["obs"]
        raw_acts = traj["acts"]

        if len(raw_acts) < 10:
            continue # skip really short runs

        aligned_obs = raw_obs
        aligned_acts = raw_acts

        # savitzky-golay filter, only on xyz (not gripper)
        window_len = min(11, len(aligned_acts) if len(aligned_acts) % 2 != 0 else len(aligned_acts) - 1)
        if window_len > 3:
            aligned_acts[:, :3] = savgol_filter(aligned_acts[:, :3], window_length=window_len, polyorder=3, axis=0)
 
        filtered_new_dataset.append({
            "obs": np.array(aligned_obs, dtype=np.float32),
            "acts": np.array(aligned_acts, dtype=np.float32),
            "terminal": traj["terminal"],
        })

    if len(filtered_new_dataset) > 3:
        filtered_new_dataset.sort(key=lambda x: len(x["acts"]))
        keep_count = int(len(filtered_new_dataset) * 0.75)
        removed = len(filtered_new_dataset) - keep_count
        filtered_new_dataset = filtered_new_dataset[:keep_count]
        print(f"Discarded {removed} meandering/slow episodes.")

    final_dataset = existing_dataset + filtered_new_dataset
    if len(new_dataset) == 0:
        print("\nNo new episodes completed. Saving existing data.")
        final_dataset = existing_dataset
 
    with open(output_file, "wb") as f:
        pickle.dump(final_dataset, f)
    print(f"Filtered Dataset saved successfully with {len(final_dataset)} optimized episodes.")

# shared recording loop for all input devices/methods
def run_operator_session(input_source: InputSource, task_name: str, total_episodes: int, output_file: str):
    # note that task_name and act_dim are defined in TASK_CONFIG
    cfg = TASK_CONFIG[task_name]
    env_id = cfg["env_id"]
    act_dim = cfg["act_dim"]
 
    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                existing_dataset = pickle.load(f)
            print(f"Loaded {len(existing_dataset)} existing episodes from {output_file}.")
        except Exception:
            existing_dataset = []
    else:
        existing_dataset = []
 
    initial_count = len(existing_dataset)
    new_dataset = []

    # make, flatten, and smooth gymnasium environment
    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    flat_env = FlattenGoalEnv(raw_env)
    env = SmoothFrankaWrapper(flat_env, dt=CONTROL_DT)
 
    # make sure that we got the right act_dim from TASK_CONFIG, just in case to avoid accidental incorrect task
    env_act_dim = env.action_space.shape[0]
    if env_act_dim != act_dim:
        raise ValueError(
            f"TASK_CONFIG says act_dim={act_dim} for '{task_name}' but "
            f"{env_id}'s action_space is {env_act_dim}-D. Check smooth_env.py / TASK_CONFIG."
        )
 
    env_ready_hook_fired = False
 
    try:
        ep = 0
        while ep < total_episodes:
            obs, info = env.reset()
 
            if not env_ready_hook_fired:
                input_source.on_env_ready(env)
                env_ready_hook_fired = True
 
            input_source.reset_state()
 
            done = False
            aborted = False
            terminated = False
            truncated = False
 
            ep_obs = [obs]
            ep_acts = []
 
            current_ep_num = initial_count + ep + 1
            print(f"\n--- Recording Episode {current_ep_num}/{initial_count + total_episodes} ---")
 
            next_tick = time.perf_counter()
 
            while not done:
                if input_source.should_abort():
                    print(f"Episode {current_ep_num} aborted by operator.")
                    aborted = True
                    break

                # get raw input from device, and needed action (now clipped vector)
                raw_action = input_source.get_action(act_dim)
                raw_action = np.clip(raw_action, env.action_space.low, env.action_space.high)

                next_obs, reward, terminated, truncated, info = env.step(raw_action)
                done = terminated or truncated

                ep_obs.append(next_obs)
                ep_acts.append(raw_action)
                obs = next_obs

                next_tick += CONTROL_DT
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
            
            if aborted:
                print(f"Trial {current_ep_num} discarded (cancelled by operator).")
            elif truncated:
                print(f"Trial {current_ep_num} discarded (reached max episode steps).")
            elif terminated:
                new_dataset.append({
                    "obs": np.array(ep_obs, dtype=np.float32),
                    "acts": np.array(ep_acts, dtype=np.float32),
                    "terminal": True,
                })
                print(f"Saved Episode {current_ep_num} with {len(ep_acts)} steps.")
                ep += 1
            else:
                print(f"Trial {current_ep_num} discarded (ended w/o success).")
 
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
 
    finally:
        print("Closing Gymnasium environment...")
        input_source.close()
        env.close()
        _filter_and_save(existing_dataset, new_dataset, output_file)

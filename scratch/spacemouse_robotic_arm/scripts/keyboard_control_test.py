import time
import numpy as np
import gymnasium as gym
import panda_mujoco_gym
from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper
from keyboard_teleop import KeyboardTeleop

def disable_mujoco_key_callbacks(env):
    # method to disable default key bindings in mujoco so we can use it for keyboard control
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "mujoco_renderer"):
            viewer = unwrapped.mujoco_renderer._get_viewer("human")
            if hasattr(viewer, "window") and viewer.window is not None:
                import glfw
                glfw.set_key_callback(viewer.window, lambda window, key, scancode, action, mods: None)
                print("[INFO] Native MuJoCo key bindings successfully disabled.")
    except Exception as e:
        print(f"[WARNING] Could not disable MuJoCo key callbacks: {e}")

def main():
    # Prompt for task selection
    print("Select Task Environment to Test:")
    print("1: Franka Push (3D Action Space: X, Y, Z)")
    print("2: Franka Pick & Place (4D Action Space: X, Y, Z, Gripper)")
    task_num = int(input("Enter task number (1 or 2): "))

    if task_num == 1:
        env_id = "FrankaPushSparse-v0"
    else:
        env_id = "FrankaPickAndPlaceSparse-v0"

    print(f"\nInitializing 3D Simulation for {env_id}...")
    
    # Instantiate raw environment and apply wrappers
    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    disable_mujoco_key_callbacks(raw_env)
    flat_env = FlattenGoalEnv(raw_env)
    env = SmoothFrankaWrapper(flat_env)

    # initialize teleop
    teleop = KeyboardTeleop(move_speed=0.4, ramp_rate=0.15)

    obs, info = env.reset()
    done = False
    step_count = 0

    try:
        while not done:
            # Get 4D action from keyboard teleop [x, y, z, gripper]
            action = teleop.get_action()

            # For 3D tasks (Push), slice out the 4th element if present
            if task_num == 1 and len(action) > 3:
                action = action[:3]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            # Print action readouts every 20 steps (~0.4s) to monitor bounds
            if step_count % 20 == 0:
                act_str = ", ".join([f"{val:+.2f}" for val in action])
                print(f"Step {step_count:4d} | Executed Action: [{act_str}]")

            # Maintain ~50Hz rendering frequency
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    finally:
        print("\nCleaning up environment and display...")
        teleop.close()
        env.close()
        print("Test completed successfully.")


if __name__ == "__main__":
    main()
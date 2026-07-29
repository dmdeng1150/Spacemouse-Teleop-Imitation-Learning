import gymnasium as gym
import numpy as np
import time
import os
import panda_mujoco_gym

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from training_code import FlattenGoalEnv  # Import your observation wrapper

def test_saved_model(model_path="bc_panda_spacemouse_model.pt", env_id="FrankaPickAndPlaceSparse-v0", num_episodes=5):
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return

    print(f"Loading model from '{model_path}'...")

    # 1. Load the policy based on model type
    if model_path.endswith(".pt"):
        # Load Behavioral Cloning (BC) Policy
        policy = ActorCriticPolicy.load(model_path)
    else:
        # Load PPO / AIRL Model (.zip)
        model = PPO.load(model_path)
        policy = model.policy

    print("Model loaded successfully!")

    # 2. Create evaluation environment with 3D GUI enabled
    print("\n--- Launching 3D Simulation Evaluation ---")
    raw_env = gym.make(env_id, render_mode="human", max_episode_steps=1000)
    env = FlattenGoalEnv(raw_env)

    success_count = 0

    # 3. Run evaluation episodes
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_success = False

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        while not done:
            # Predict action deterministically from trained policy network
            action, _ = policy.predict(obs, deterministic=True)

            # Step environment forward
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if info.get("is_success", False) or terminated:
                ep_success = True

            # Pace loop so 3D GUI renders smoothly (~50 FPS)
            time.sleep(0.02)

        if ep_success:
            success_count += 1
            print(f"Result: SUCCESS in {step} steps!")
        else:
            print(f"Result: FAILED (reached max steps).")

    env.close()

    # Print summary results
    success_rate = (success_count / num_episodes) * 100.0
    print(f"\n==========================================")
    print(f"Final Evaluation Summary:")
    print(f"   Successful Runs: {success_count}/{num_episodes}")
    print(f"   Success Rate:    {success_rate:.1f}%")
    print(f"==========================================")

if __name__ == "__main__":
    # To test Behavioral Cloning (BC) model:
    test_saved_model("bc_panda_spacemouse_model.pt", env_id="FrankaPickAndPlaceSparse-v0")
    
    # To test AIRL model (uncomment line below):
    # test_saved_model("airl_panda_spacemouse_model.zip", env_id="FrankaPickAndPlaceSparse-v0")
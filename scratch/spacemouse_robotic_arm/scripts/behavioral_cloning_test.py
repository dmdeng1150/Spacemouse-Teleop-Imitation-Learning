import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time
import matplotlib.pyplot as plt

from imitation.data import types
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper
from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper


def evaluate_success_rate(policy, env_id="FrankaPickAndPlaceSparse-v0", num_episodes=10, max_steps=350):
    eval_raw = gym.make(env_id, max_episode_steps=max_steps)
    eval_env_flat = FlattenGoalEnv(eval_raw)
    eval_env = SmoothFrankaWrapper(eval_env_flat)
    
    successes = 0
    for _ in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_success = False
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            
            # snap gripper prediction to binary
            action[3] = 1.0 if action[3] >= 0.0 else -1.0
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            if info.get("is_success", False) or terminated:
                ep_success = True
                
        if ep_success:
            successes += 1
            
    eval_env.close()
    return (successes / num_episodes) * 100.0


def evaluate_and_view_policy(policy, env_id="FrankaPickAndPlaceSparse-v0", num_episodes=3, max_steps=350):
    print("\n--- Launching 3D Simulation Evaluation ---")
    eval_raw = gym.make(env_id, render_mode="human", max_episode_steps=max_steps)
    eval_env_flat = FlattenGoalEnv(eval_raw)
    eval_env = SmoothFrankaWrapper(eval_env_flat)
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        print(f"Running Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            
            # snap continuous gripper prediction to binary state, since that's how we train it
            action[3] = 1.0 if action[3] >= 0.0 else -1.0
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            time.sleep(0.02)
            
        print(f"Episode {ep + 1} ended after {step} steps.")
        
    eval_env.close()


def plot_success_rate(epochs, success_rates, save_path="bc_success_rate.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, success_rates, marker='o', linewidth=2, color='#1f77b4', label='BC Policy Success Rate')
    
    plt.title('Behavioral Cloning: Success Rate vs. Training Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    
    plt.ylim(-5, 105)
    plt.xlim(1, max(epochs))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully as '{save_path}'.")
    plt.show()

def train_on_operator_data(data_path="operator_data.pkl", total_epochs=20, eval_freq=4, eval_episodes=5):
    env_id = "FrankaPickAndPlaceSparse-v0"
    
    # Instantiate wrapped environment matching demonstration specs
    raw_env = gym.make(env_id, max_episode_steps=350)
    flat_env = FlattenGoalEnv(raw_env)
    smooth_env = SmoothFrankaWrapper(flat_env)
    train_env = RolloutInfoWrapper(smooth_env)
    
    # Load and parse raw pickled trajectories
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    formatted_dataset = []
    MIN_STEPS_THRESHOLD = 10
    
    for traj in raw_trajectories:
        if len(traj["acts"]) < MIN_STEPS_THRESHOLD:
            continue
        dummy_rewards = np.zeros(len(traj["acts"]), dtype=np.float32)
        formatted_dataset.append(
            types.TrajectoryWithRew(
                obs=traj["obs"],
                acts=traj["acts"],
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rewards
            )
        )
        
    print(f"Loaded {len(formatted_dataset)} valid operator runs for training.")

    # Instantiate BC Trainer with L2 regularization
    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=formatted_dataset,
        l2_weight=1e-4,
        rng=np.random.default_rng(seed=42)
    )
    
    # Train Epoch-by-Epoch & Track Success Rate
    print("\nStarting Behavioral Cloning with Success Rate Tracking...")
    epochs_list = []
    success_rates_list = []

    # Evaluate baseline before training (Epoch 0)
    initial_rate = evaluate_success_rate(bc_trainer.policy, env_id=env_id, num_episodes=eval_episodes_per_epoch)
    print(f"Epoch  0/{total_epochs:2d} | Baseline Success Rate: {initial_rate:5.1f}%")

    for epoch in range(1, total_epochs + 1):
        # Train 1 epoch at a time
        bc_trainer.train(n_epochs=1)
        
        if epoch % eval_freq == 0 or epoch == total_epochs:
            success_rate = evaluate_success_rate(bc_trainer.policy, env_id=env_id, num_episodes=eval_episodes)
            epochs_list.append(epoch)
            success_rates_list.append(success_rate)
            print(f"Epoch {epoch:2d}/{total_epochs:2d} | Success Rate: {success_rate:5.1f}%")
        else:
            print(f"Epoch {epoch:2d}/{total_epochs:2d} | Training...")

    # save trained policy as pytorch model
    bc_trainer.policy.save("bc_panda_spacemouse_model.pt")
    print("\nModel saved successfully as 'bc_panda_spacemouse_model.pt'")

    # plot success rate vs epochs
    plot_success_rate(epochs_list, success_rates_list)

    # visual evaluation of robot movement
    evaluate_and_view_policy(bc_trainer.policy, env_id=env_id, num_episodes=3)


if __name__ == "__main__":
    train_on_operator_data(total_epochs=40, eval_freq=5, eval_episodes=5)
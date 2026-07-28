import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time

from imitation.data import types
from imitation.algorithms import bc
from imitation.data.wrappers import RolloutInfoWrapper
from training_code import FlattenGoalEnv # Import data from demos

def evaluate_and_view_policy(policy, num_episodes=3):
    """Opens the MuJoCo viewer and watches the trained BC agent perform the task live."""
    print("\n--- Launching 3D Simulation Evaluation ---")
    
    # 1. Create evaluation environment with human rendering enabled
    eval_raw = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human", max_episode_steps = 350)
    eval_env = FlattenGoalEnv(eval_raw)
    
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        
        print(f"Running Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            # Predict action from the trained policy network
            action, _ = policy.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            
            # Pacing so the 3D GUI updates smoothly
            time.sleep(0.02)
            
        print(f"Episode {ep + 1} ended after {step} steps.")
        
    eval_env.close()

def train_on_operator_data(data_path="operator_data.pkl"):
    # 1. Instantiate wrapped environment matching data dimensionality
    raw_env = gym.make("FrankaPickAndPlaceSparse-v0")
    flat_env = FlattenGoalEnv(raw_env)
    train_env = RolloutInfoWrapper(flat_env) # Required for tracking internal imitation steps
    
    # 2. Load and parse the raw pickled operator trajectories
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    formatted_dataset = []
    for traj in raw_trajectories:
        # Imitation framework expects a dummy reward array corresponding to action shapes
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
        
    print(f"Loaded {len(formatted_dataset)} operator runs for training.")
    

    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=formatted_dataset,
        rng=np.random.default_rng(seed=42)
    )
    
    # 4. Train policy network
    print("Starting Behavioral Cloning on your SpaceMouse runs...")
    bc_trainer.train(n_epochs=20)
    
    # 5. Save the trained weights
    bc_trainer.policy.save("bc_panda_spacemouse_model.pt")
    print("Model saved successfully as 'bc_panda_spacemouse_model.pt'")

    evaluate_and_view_policy(bc_trainer.policy, num_episodes=3)

if __name__ == "__main__":
    train_on_operator_data()

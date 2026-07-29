import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time

from imitation.data import types
from imitation.algorithms import bc  
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper # import wrapper from demos

def evaluate_and_view_policy(policy, num_episodes=3):
    """Opens the MuJoCo viewer and watches the trained AIRL agent perform the task live."""
    print("\n--- Launching 3D Simulation Evaluation ---")
    
    # 1. Create evaluation environment with human rendering enabled
    eval_raw = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human", max_episode_steps=350)
    eval_env_flat = FlattenGoalEnv(eval_raw)
    eval_env = SmoothFrankaWrapper(eval_env_flat)
    
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
    # 1. Instantiate Vectorized Environment (AIRL requires VecEnv for online rollouts)
    def make_env():
        raw_env = gym.make("FrankaPickAndPlaceSparse-v0", max_episode_steps=1000)
        flat_env = FlattenGoalEnv(raw_env)
        smooth_env = SmoothFrankaWrapper(flat_env)
        return RolloutInfoWrapper(smooth_env)

    venv = DummyVecEnv([make_env])
    
    # 2. Load and parse the raw pickled operator trajectories
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    formatted_dataset = []
    for traj in raw_trajectories:
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

    # 3. Define the Generative RL Agent (PPO) that learns via AIRL rewards
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.01,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        n_epochs=10,
        seed=42
    )
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=learner.policy,  # <-- In-place updates learner.policy directly!
        demonstrations=formatted_dataset,
        rng=np.random.default_rng(42),
    )
    # Train BC offline for 30 epochs (takes ~5-10 seconds on CPU/GPU)
    bc_trainer.train(n_epochs=30)
    # 4. Define the AIRL Reward Network / Discriminator Architecture
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    # 5. Build AIRL Trainer
    airl_trainer = AIRL(
        demonstrations=formatted_dataset,
        demo_batch_size=128,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )
    
    # 6. Train policy network online via AIRL
    print("Starting AIRL training on your SpaceMouse runs...")
    airl_trainer.train(total_timesteps=25_000)
    
    # 7. Save the trained generator model
    airl_trainer.gen_algo.save("airl_panda_spacemouse_model")
    print("Model saved successfully as 'airl_panda_spacemouse_model.zip'")

    # 8. Evaluate policy using the learned policy network
    evaluate_and_view_policy(airl_trainer.gen_algo.policy, num_episodes=3)

if __name__ == "__main__":
    train_on_operator_data()
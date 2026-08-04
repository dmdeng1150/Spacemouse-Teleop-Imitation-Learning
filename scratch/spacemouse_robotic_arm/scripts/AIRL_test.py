import gymnasium as gym
import pickle
import numpy as np
import panda_mujoco_gym
import time
import os
import matplotlib.pyplot as plt

from imitation.data import types
from imitation.algorithms import bc  
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.data.wrappers import RolloutInfoWrapper

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import KVWriter

from smooth_env import FlattenGoalEnv, SmoothFrankaWrapper, RelativeGoalWrapper, BinaryGripperActionWrapper, FixDoneWrapper, SmoothXYZActionWrapper

# --- CUSTOM LOGGER ---
class CustomLogCollector(KVWriter):
    def __init__(self):
        self.logs = []
    def write(self, key_values, key_excluded, step=0):
        self.logs.append(key_values.copy())
    def close(self):
        pass

def evaluate_and_view_policy(policy, num_episodes=5):
    print("\n--- Launching 3D Simulation Evaluation ---")
    
    eval_raw = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human", max_episode_steps=250)
    eval_env_flat = FlattenGoalEnv(eval_raw)
    eval_env_smooth = SmoothFrankaWrapper(eval_env_flat)
    eval_env_rel = RelativeGoalWrapper(eval_env_smooth)
    eval_env = FixDoneWrapper(eval_env_rel)
    
    successes = 0
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        done = False
        step = 0
        
        print(f"Running Evaluation Episode {ep + 1}/{num_episodes}...")
        
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            action = np.array(action, copy=True).flatten()
            
            # Snap gripper for evaluation
            if len(action) > 3:
                action[3] = 1.0 if action[3] > 0.1 else -1.0

            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            step += 1
            time.sleep(0.01)
            
            if info.get("is_success", False):
                successes += 1

        print(f"Episode {ep + 1} ended after {step} steps. Success: {info.get('is_success', False)}")
        
    print(f"\nFinal Visual Evaluation Success Rate: {(successes / num_episodes) * 100:.1f}%")
    eval_env.close()


def train_on_operator_data(data_path="operator_data_pick_and_place_spacemouse.pkl"):
    num_cpu = os.cpu_count() or 8  
    
    def make_env(rank, seed=0):
        def _init():
            raw_env = gym.make("FrankaPickAndPlaceSparse-v0", max_episode_steps=250)
            raw_env.action_space.seed(seed + rank)
            flat_env = FlattenGoalEnv(raw_env)
            smooth_env = SmoothFrankaWrapper(flat_env, dt=0.01)
            grip_env = BinaryGripperActionWrapper(smooth_env, close_thresh=0.2, open_thresh=0.6)
            rel_env = RelativeGoalWrapper(grip_env)            
            fixed_env = FixDoneWrapper(rel_env) 
            return RolloutInfoWrapper(fixed_env)
        return _init

    print(f"Launching {num_cpu} parallel physics simulations...")
    venv = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    with open(data_path, "rb") as f:
        raw_trajectories = pickle.load(f)
        
    # Apply RelativeGoalWrapper transformation to demonstration observations as well!
    # Create a temporary single env to process trajectory observations
    sample_env = gym.make("FrankaPickAndPlaceSparse-v0")
    sample_flat = FlattenGoalEnv(sample_env)
    sample_franka = SmoothFrankaWrapper(sample_flat)
    sample_xyz = SmoothXYZActionWrapper(sample_franka)
    sample_grip = BinaryGripperActionWrapper(sample_xyz)
    rel_transformer = RelativeGoalWrapper(sample_grip, ee_z_offset=0.058)

    formatted_dataset = []
    for traj in raw_trajectories:
        dummy_rewards = np.zeros(len(traj["acts"]), dtype=np.float32)
        
        # Transform raw observations to include relative features
        transformed_obs = np.array([rel_transformer.observation(o) for o in traj["obs"]], dtype=np.float32)

        formatted_dataset.append(
            types.TrajectoryWithRew(
                obs=transformed_obs,
                acts=traj["acts"],
                infos=None,
                terminal=traj["terminal"],
                rews=dummy_rewards
            )
        )
    sample_env.close()
        
    print(f"Loaded {len(formatted_dataset)} operator runs for training.")

    total_expert_transitions = sum(len(traj.acts) for traj in formatted_dataset)
    safe_batch_size = min(1024, total_expert_transitions)

    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,               
        n_steps=2048 // num_cpu,      
        ent_coef=0.02,                
        learning_rate=5e-5,           
        gamma=0.99,
        clip_range=0.05,               
        n_epochs=10,                  
        seed=42,
        device="cpu"                 
    )
    
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=learner.policy,
        demonstrations=formatted_dataset,
        rng=np.random.default_rng(42),
    )
    
    log_collector = CustomLogCollector()
    sb3_logger = getattr(bc_trainer.logger, "default_logger", bc_trainer.logger)
    sb3_logger.output_formats.append(log_collector)
    
    print("\nPre-training Generator via Behavioral Cloning (Offline)...")
    bc_epochs = 100
    bc_trainer.train(n_epochs=bc_epochs)
    
    loss_list = []
    prob_true_act_list = []
    for log in log_collector.logs:
        loss = log.get("bc/loss", log.get("training/loss", log.get("loss", None)))
        prob = log.get("bc/prob_true_act", log.get("training/prob_true_act", log.get("prob_true_act", None)))
        if loss is not None and prob is not None:
            loss_list.append(loss)
            prob_true_act_list.append(prob)
            
    tracked_epochs = list(range(1, len(loss_list) + 1))
    plot_training_metrics(tracked_epochs, prob_true_act_list, loss_list, save_path="bc_pretraining_metrics.png")
    
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=None,
    )

    airl_trainer = AIRL(
        demonstrations=formatted_dataset,
        demo_batch_size=safe_batch_size,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,   
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )
    
    print("\nStarting AIRL training (Online)...")
    airl_trainer.train(total_timesteps=100_000)
    
    venv.close()
    
    airl_trainer.gen_algo.save("airl_panda_spacemouse_model")
    print("\nModel saved successfully as 'airl_panda_spacemouse_model.zip'")

    evaluate_and_view_policy(airl_trainer.gen_algo.policy, num_episodes=5)


if __name__ == "__main__":
    train_on_operator_data()
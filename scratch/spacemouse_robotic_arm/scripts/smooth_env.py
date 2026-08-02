import gymnasium as gym
import numpy as np

class FlattenGoalEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = env.observation_space.spaces
        low = np.concatenate([spaces['observation'].low, spaces['achieved_goal'].low, spaces['desired_goal'].low])
        high = np.concatenate([spaces['observation'].high, spaces['achieved_goal'].high, spaces['desired_goal'].high])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs['observation'], obs['achieved_goal'], obs['desired_goal']])

class SmoothFrankaWrapper(gym.ActionWrapper):
    """
    Wraps Franka Mujoco Gym to integrate smoothed displacement accumulation 
    and soft-hold joint filtering directly into the environment dynamics.
    Dynamically handles both 3D (Push) and 4D (Pick and Place) action spaces.
    """
    def __init__(self, env, sensitivity=0.25, max_accel=6.0, dt=1.0/100.0, ema_beta=0.05):
        super().__init__(env)
        self.sensitivity = sensitivity
        self.max_accel = max_accel
        self.dt = dt
        self.ema_beta = ema_beta
        
        self.current_action_xyz = np.zeros(3, dtype=np.float32)
        self.ctrl_target = None

    def reset(self, **kwargs):
        self.current_action_xyz = np.zeros(3, dtype=np.float32)
        self.ctrl_target = None
        return self.env.reset(**kwargs)

    def action(self, raw_action):
        """Processes raw SpaceMouse / Agent actions through slew-rate limiter."""
        desired_xyz = raw_action[:3] * self.sensitivity

        # Slew-rate limiting (limits acceleration)
        action_change = desired_xyz - self.current_action_xyz
        max_change = self.max_accel * self.dt
        action_change = np.clip(action_change, -max_change, max_change)
        
        self.current_action_xyz += action_change
        xyz_action = np.clip(self.current_action_xyz, -1.0, 1.0)

        # Check if the environment provides a gripper command (4D) or not (3D)
        if len(raw_action) == 4:
            gripper = raw_action[3]
            processed_action = np.array([
                xyz_action[1],
                xyz_action[0],
                xyz_action[2],
                gripper
            ], dtype=np.float32)
        else:
            # 3D Action Space (e.g., Push Task - NO GRIPPER)
            processed_action = np.array([
                xyz_action[1],
                xyz_action[0],
                xyz_action[2]
            ], dtype=np.float32)

        return processed_action

    def step(self, action):
        # Apply the soft-hold joint target trick during physics step
        franka_env = self.env.unwrapped
        current_q = np.array([
            franka_env._utils.get_joint_qpos(franka_env.model, franka_env.data, name) 
            for name in franka_env.arm_joint_names
        ]).flatten()

        if self.ctrl_target is None:
            self.ctrl_target = current_q.copy()
        else:
            self.ctrl_target = (1 - self.ema_beta) * self.ctrl_target + self.ema_beta * current_q
            franka_env.data.ctrl[0:7] = self.ctrl_target

        return self.env.step(self.action(action))

# --- FIXED WRAPPER: RELATIVE OBSERVATION FEATURES ---
class RelativeGoalWrapper(gym.ObservationWrapper):
    """Appends relative displacement vectors (ee_to_block, block_to_goal) to observation.
       Uses exact FlattenGoalEnv layout: achieved_goal is obs[-6:-3], desired_goal is obs[-3:].
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape[0]
        new_shape = old_shape + 6
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

    def observation(self, obs):
        ee_pos = obs[0:3]          # First 3 elements: End-Effector 3D Position
        block_pos = obs[-6:-3]     # FIX: Achieved Goal (Block 3D Position)
        goal_pos = obs[-3:]        # Last 3 elements: Desired Goal 3D Position

        rel_ee_to_block = block_pos - ee_pos
        rel_block_to_goal = goal_pos - block_pos

        return np.concatenate([obs, rel_ee_to_block, rel_block_to_goal]).astype(np.float32)


# --- WRAPPER 2: BOOLEAN SIGNAL FIX ---
class FixDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info
    
class BinaryGripperActionWrapper(gym.ActionWrapper):
    """Snaps the gripper action index 3 so PPO exploration noise doesn't chatter the fingers."""
    def action(self, action):
        action = np.array(action, copy=True)
        if len(action) > 3:
            action[3] = 1.0 if action[3] > 0.1 else -1.0
        return action
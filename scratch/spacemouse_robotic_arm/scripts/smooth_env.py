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
    Wraps Franka Mujoco Gym to integrate smoothed displacement accumulation.
    Dynamically handles both 3D (Push) and 4D (Pick and Place) action spaces.
    """
    def __init__(self, env, sensitivity=0.15, max_accel=1.5, dt=1.0/100.0):
        super().__init__(env)
        self.sensitivity = sensitivity
        self.max_accel = max_accel
        self.dt = dt
        self.current_action_xyz = np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        self.current_action_xyz = np.zeros(3, dtype=np.float32)
        return self.env.reset(**kwargs)

    def action(self, raw_action):
        """Processes raw SpaceMouse / Agent actions through slew-rate limiter."""
        desired_xyz = raw_action[:3] * self.sensitivity

        # Slew-rate limiting (limits acceleration for smooth physics)
        action_change = desired_xyz - self.current_action_xyz
        max_change = self.max_accel * self.dt
        action_change = np.clip(action_change, -max_change, max_change)
        
        self.current_action_xyz += action_change
        xyz_action = np.clip(self.current_action_xyz, -1.0, 1.0)

        # Handle 4D (Pick & Place) vs 3D (Push)
        if len(raw_action) == 4:
            gripper = raw_action[3]
            processed_action = np.array([
                xyz_action[1],
                xyz_action[0],
                xyz_action[2],
                gripper
            ], dtype=np.float32)
        else:
            processed_action = np.array([
                xyz_action[1],
                xyz_action[0],
                xyz_action[2]
            ], dtype=np.float32)

        return processed_action

    def step(self, action):
        return self.env.step(self.action(action))

# --- DYNAMIC FUNNEL DESCENT OBSERVATION WRAPPER ---
class RelativeGoalWrapper(gym.ObservationWrapper):
    """Appends Smooth Funnel Approach Features:
       1. Dynamic Funnel Target Vector (rel_ee_to_funnel) [3 features]
       2. Direct 3D Relative Vectors (block - ee, goal - block) [6 features]
       3. Unit 2D XY Direction Vector (dir_xy) [2 features]
       4. Scalar Distances (dist_xy, dist_block_goal) [2 features]
       5. Smooth Alignment Factor exp(-dist_xy / 0.03) [1 feature]
       6. Block Height Z [1 feature]
       7. Current Latched Gripper State (-1.0 or 1.0) [1 feature]
       Total added features = 16
    """
    def __init__(self, env, max_clearance=0.05, transition_dist=0.08):
        super().__init__(env)
        self.max_clearance = max_clearance        
        self.transition_dist = transition_dist    
        
        old_shape = env.observation_space.shape[0]
        new_shape = old_shape + 16                # 16 engineered features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

    def _get_gripper_state(self, override_gripper=None):
        """Fetches current latched gripper state, or uses override during dataset transformation."""
        if override_gripper is not None:
            return np.array([override_gripper], dtype=np.float32)
            
        env = self.env
        while hasattr(env, 'env'):
            if isinstance(env, BinaryGripperActionWrapper):
                return np.array([env.state], dtype=np.float32)
            env = env.env
        return np.array([1.0], dtype=np.float32)

    def observation(self, obs, override_gripper=None):
        ee_pos = obs[0:3]          # First 3 elements: End-Effector 3D Position
        block_pos = obs[-6:-3]     # Achieved Goal (Block 3D Position)
        goal_pos = obs[-3:]        # Desired Goal (Target 3D Position)

        # 1. 2D Horizontal Distance & Direction
        rel_xy = block_pos[:2] - ee_pos[:2]
        dist_xy_val = np.linalg.norm(rel_xy)
        dir_xy = rel_xy / (dist_xy_val + 1e-6)
        dist_xy = np.array([dist_xy_val], dtype=np.float32)

        # 2. Dynamic Funnel Target Position
        clearance_ratio = min(1.0, dist_xy_val / self.transition_dist)
        funnel_target_z = block_pos[2] + self.max_clearance * clearance_ratio
        
        funnel_target_pos = np.array([block_pos[0], block_pos[1], funnel_target_z], dtype=np.float32)
        rel_ee_to_funnel = funnel_target_pos - ee_pos

        # 3. Direct 3D Relative Displacement Vectors
        rel_ee_to_block = block_pos - ee_pos
        rel_block_to_goal = goal_pos - block_pos

        # 4. Smooth Continuous Alignment Factor
        smooth_alignment = np.array([np.exp(-dist_xy_val / 0.03)], dtype=np.float32)

        # 5. Goal Distance & Block Height Z
        dist_block_goal = np.array([np.linalg.norm(rel_block_to_goal)], dtype=np.float32)
        block_height = np.array([block_pos[2]], dtype=np.float32)

        # 6. Current Latched Gripper State (-1.0 or 1.0)
        gripper_state = self._get_gripper_state(override_gripper=override_gripper)

        return np.concatenate([
            obs, 
            rel_ee_to_funnel,
            rel_ee_to_block, 
            rel_block_to_goal, 
            dir_xy,
            dist_xy,
            smooth_alignment,
            dist_block_goal, 
            block_height,
            gripper_state
        ]).astype(np.float32)
    
# --- WRAPPER 2: BOOLEAN SIGNAL FIX ---
class FixDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info
    
class SmoothXYZActionWrapper(gym.ActionWrapper):
    """Applies Exponential Moving Average (EMA) ONLY to XYZ movement (indices 0, 1, 2).
       Prevents spatial drift without corrupting gripper signals.
    """
    def __init__(self, env, alpha=0.65):
        super().__init__(env)
        self.alpha = alpha 
        self.last_xyz = np.zeros(3, dtype=np.float32)

    def reset(self, **kwargs):
        self.last_xyz = np.zeros(3, dtype=np.float32)
        return super().reset(**kwargs)

    def action(self, action):
        action = np.array(action, copy=True)
        if len(action) >= 3:
            smoothed_xyz = (1.0 - self.alpha) * action[:3] + self.alpha * self.last_xyz
            self.last_xyz = smoothed_xyz
            action[:3] = smoothed_xyz
        return action
    
class BinaryGripperActionWrapper(gym.ActionWrapper):
    """Hysteresis Gripper Latching.
       Locks the gripper state until a strong reverse command is received.
       Flickering/chattering is mathematically impossible with this wrapper.
    """
    def __init__(self, env, close_thresh=0.2, open_thresh=0.6):
        super().__init__(env)
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.state = 1.0  # Start OPEN

    def reset(self, **kwargs):
        self.state = 1.0  # Reset to OPEN on episode start
        return super().reset(**kwargs)

    def action(self, action):
        action = np.array(action, copy=True)
        if len(action) > 3:
            raw_g = action[3]
            
            # Hysteresis Bounds
            if raw_g < self.close_thresh:
                self.state = -1.0  # Lock CLOSED
            elif raw_g > self.open_thresh:
                self.state = 1.0   # Lock OPEN
            # If raw_g is between -0.1 and 0.6, KEEP PREVIOUS STATE!

            action[3] = self.state
        return action
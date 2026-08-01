import numpy as np
from pynput import keyboard

class KeyboardTeleop:
    """
    key input listener using pynput
    """
    def __init__(self, move_speed=0.4, speed_boost=2.0, ramp_rate=0.15):
        self.move_speed = move_speed
        self.speed_boost = speed_boost
        self.ramp_rate = ramp_rate

        self.vel = np.zeros(3, dtype=np.float32)
        
        # toggle state (false is open, -1.0, and true is closed, 1.0)
        self.is_closed = False

        # track what keys are pressed currently
        self.pressed_keys = set()

        # start nonblocking os-level kb listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            # read character keys
            self.pressed_keys.add(key.char.lower())
        except AttributeError:
            # read special keys
            if key in (keyboard.Key.shift, keyboard.Key.shift_r):
                self.pressed_keys.add("shift")
            elif key == keyboard.Key.space:
                # toggle gripper state on press
                self.is_closed = not self.is_closed

    def _on_release(self, key):
        try:
            self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            if key in (keyboard.Key.shift, keyboard.Key.shift_r):
                self.pressed_keys.discard("shift")

    def get_action(self):
        # calculate x, y, z, gripper
        speed_mult = self.speed_boost if "shift" in self.pressed_keys else 1.0
        target_speed = self.move_speed * speed_mult

        target_vel = np.zeros(3, dtype=np.float32)

        # directional mapping
        if 'a' in self.pressed_keys: target_vel[0] -= target_speed  # Left (-X)
        if 'd' in self.pressed_keys: target_vel[0] += target_speed  # Right (+X)
        if 'w' in self.pressed_keys: target_vel[1] += target_speed  # Forward (+Y)
        if 's' in self.pressed_keys: target_vel[1] -= target_speed  # Backward (-Y)
        if 'e' in self.pressed_keys: target_vel[2] += target_speed  # Up (+Z)
        if 'q' in self.pressed_keys: target_vel[2] -= target_speed  # Down (-Z)

        # toggle state of gripper
        gripper_state = 1.0 if self.is_closed else -1.0

        # smooth velocity ramping for translational axes
        self.vel = (1.0 - self.ramp_rate) * self.vel + self.ramp_rate * target_vel

        return np.append(self.vel, gripper_state)

    def close(self):
        self.listener.stop()
"""
Thermal-Conditioned Lift Environment — v3 (State + Optional Image).

Observation space depends on `use_images`:

  use_images=False  →  Box(24,)  flat state:
      [joint_pos (7) | joint_temps (7) | eef_pos (3) | eef_quat (4) | cube_pos (3)]
      Policy receives explicit cube position so it can reach/grasp without vision.

  use_images=True  →  Dict:
      "state"  Box(21,): [joint_pos (7) | joint_temps (7) | eef_pos (3) | eef_quat (4)]
               cube_pos is DROPPED — the policy locates the cube from the image.
      "image"  Box(0, 255, (3, H, W), uint8): RGB from wrist eye-in-hand camera.

Action space:
  Box(-1, 1, shape=(7,)) — OSC_POSE: [dx, dy, dz, dRoll, dPitch, dYaw, gripper]

Reward (with lift_bonus fix):
  Dense shaping from robosuite (reaching → grasping).
  + CONTINUOUS lift-height reward when cube is grasped and rising (bridges the
    gap between grasp and success that robosuite's shaping misses).
  + One-time milestone bonuses at 1cm, 2cm, 3cm above table.
  + One-time success_bonus when cube first crosses the success threshold.
  + Episode does NOT terminate on success (by default) so the policy is
    incentivised to lift early rather than stall at grasp.
  NO explicit temperature penalty — the robot discovers degradation through physics.

Thermal degradation:
  - Normal (20–42 °C): full capacity
  - Warm   (43–55 °C): reduced torque, slight noise
  - Hot    (56–70 °C): heavily degraded — weak, noisy, sluggish
  - Critical (>70 °C): barely moves
"""

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces

from .thermal_model import ThermalModel
from .telemetry_model import CurrentModel, VoltageModel, TelemetryModel


class ThermalLiftEnv(gym.Env):
    """
    Gym wrapper around robosuite Lift with:
      1. Continuous thermal conditioning on ALL joints
      2. Eye-in-hand camera observation (no cube_pos)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        robots="Panda",
        hot_joint_prob=0.3,
        temp_normal_range=(20.0, 42.0),
        temp_hot_range=(45.0, 80.0),
        degradation_onset=43.0,
        degradation_full=75.0,
        torque_noise_scale=0.15,
        use_images=True,
        camera_name="robot0_eye_in_hand",
        image_size=(84, 84),
        horizon=200,
        reward_scale=1.0,
        success_bonus=20.0,
        lift_bonus=100.0,
        terminate_on_success=False,
        render_mode=None,
        thermal_model=None,
        # ── Current & Voltage telemetry ──
        enable_current=False,
        enable_voltage=False,
        current_model=None,
        voltage_model=None,
        current_normal_range=(0.10, 0.50),
        current_high_range=(0.65, 1.05),
        high_current_prob=0.3,
        voltage_normal_range=(0.92, 1.0),
        voltage_low_range=(0.45, 0.88),
        low_voltage_prob=0.3,
        telemetry_model=None,
    ):
        super().__init__()

        self.hot_joint_prob = hot_joint_prob
        self.temp_normal_range = temp_normal_range
        self.temp_hot_range = temp_hot_range
        self.degradation_onset = degradation_onset
        self.degradation_full = degradation_full
        self.torque_noise_scale = torque_noise_scale
        self.use_images = use_images
        self.camera_name = camera_name
        self.image_size = image_size
        self.horizon = horizon
        self.reward_scale = reward_scale
        self.success_bonus = success_bonus
        self.lift_bonus = lift_bonus
        self.terminate_on_success = terminate_on_success
        self.render_mode = render_mode

        # ── Current & Voltage config (must be set before obs space calc) ──
        self.enable_current = enable_current
        self.enable_voltage = enable_voltage
        self.current_normal_range = current_normal_range
        self.current_high_range = current_high_range
        self.high_current_prob = high_current_prob
        self.voltage_normal_range = voltage_normal_range
        self.voltage_low_range = voltage_low_range
        self.low_voltage_prob = low_voltage_prob

        # OSC_POSE: [dx, dy, dz, droll, dpitch, dyaw, gripper] = 7 dims
        self._arm_action_dim = 6
        self._gripper_dim_start = 6

        # ── Create robosuite environment ──────────────────────────────────
        # We need offscreen renderer whenever we use camera obs OR any render
        needs_offscreen = use_images or (render_mode in ("rgb_array", "human"))
        camera_names_list = [camera_name] if use_images else []

        self.env = suite.make(
            "Lift",
            robots=robots,
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=needs_offscreen,
            use_camera_obs=use_images,
            camera_names=camera_names_list,
            camera_heights=image_size[0] if use_images else 84,
            camera_widths=image_size[1] if use_images else 84,
            reward_shaping=True,
            horizon=horizon,
        )

        # Get dimensions from a dummy reset
        obs = self.env.reset()
        # robosuite 1.4.x uses cos/sin instead of raw joint_pos
        if "robot0_joint_pos" in obs:
            self.n_joints = obs["robot0_joint_pos"].shape[0]
        else:
            self.n_joints = obs["robot0_joint_pos_cos"].shape[0]
        self.action_dim_val = self.env.action_dim

        # ── Observation space ─────────────────────────────────────────────
        # Base: joint_pos(7) + temps(7) + eef_pos(3) + eef_quat(4)
        # + optional: currents(7) + voltages(7)
        # + optional: cube_pos(3) [state-only mode]
        n_telemetry = self.n_joints  # temps always included
        if self.enable_current:
            n_telemetry += self.n_joints  # +7 for currents
        if self.enable_voltage:
            n_telemetry += self.n_joints  # +7 for voltages

        if use_images:
            state_dim = self.n_joints + n_telemetry + 3 + 4
        else:
            state_dim = self.n_joints + n_telemetry + 3 + 4 + 3

        if use_images:
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(state_dim,), dtype=np.float32,
                ),
                "image": spaces.Box(
                    low=0, high=255,
                    shape=(3, image_size[0], image_size[1]),
                    dtype=np.uint8,
                ),
            })
        else:
            # State-only: 24-dim flat vector including cube_pos
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim,), dtype=np.float32,
            )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim_val,), dtype=np.float32,
        )

        # Episode state
        self.joint_temps = np.zeros(self.n_joints, dtype=np.float32)
        self.steps = 0
        self.episode_count = 0
        self._last_obs = None

        # Lift-reward tracking (reset per episode)
        self._success_triggered = False       # one-time success bonus flag
        self._milestones_hit = set()          # height milestones already awarded
        self._table_height = 0.8              # robosuite table_offset[2]
        self._cube_rest_height = None         # set on reset (cube z at rest)

        # Image observation key in robosuite's obs dict
        self._image_obs_key = f"{camera_name}_image"

        self.joint_currents = np.zeros(self.n_joints, dtype=np.float32)
        self.joint_voltages = np.ones(self.n_joints, dtype=np.float32)

        # Thermal model — if not provided, use default linear (backward compatible)
        if thermal_model is not None:
            self.thermal_model = thermal_model
        else:
            self.thermal_model = ThermalModel.from_predefined(
                "linear",
                params={"onset": degradation_onset, "full": degradation_full, "min_cap": 0.05},
                torque_noise_scale=torque_noise_scale,
                n_joints=self.n_joints,
            )

        # Current model
        if current_model is not None:
            self.current_model = current_model
        else:
            self.current_model = CurrentModel.from_predefined(
                "linear", n_joints=self.n_joints)

        # Voltage model
        if voltage_model is not None:
            self.voltage_model = voltage_model
        else:
            self.voltage_model = VoltageModel.from_predefined(
                "linear", n_joints=self.n_joints)

        # If a combined TelemetryModel is provided, use it to override
        if telemetry_model is not None:
            self.thermal_model = telemetry_model.thermal
            self.current_model = telemetry_model.current
            self.voltage_model = telemetry_model.voltage
            self.enable_current = telemetry_model.enable_current
            self.enable_voltage = telemetry_model.enable_voltage

    # ── Telemetry sampling ──────────────────────────────────────────────

    def _sample_all_temps(self):
        """Sample temperatures for ALL joints independently."""
        self.joint_temps = np.zeros(self.n_joints, dtype=np.float32)
        for j in range(self.n_joints):
            if np.random.random() < self.hot_joint_prob:
                self.joint_temps[j] = np.random.uniform(*self.temp_hot_range)
            else:
                self.joint_temps[j] = np.random.uniform(*self.temp_normal_range)

    def _sample_all_currents(self):
        """Sample current ratios (fraction of rated) for all joints."""
        self.joint_currents = np.zeros(self.n_joints, dtype=np.float32)
        for j in range(self.n_joints):
            if np.random.random() < self.high_current_prob:
                self.joint_currents[j] = np.random.uniform(*self.current_high_range)
            else:
                self.joint_currents[j] = np.random.uniform(*self.current_normal_range)

    def _sample_all_voltages(self):
        """Sample voltage ratios (fraction of rated) for all joints."""
        self.joint_voltages = np.ones(self.n_joints, dtype=np.float32)
        for j in range(self.n_joints):
            if np.random.random() < self.low_voltage_prob:
                self.joint_voltages[j] = np.random.uniform(*self.voltage_low_range)
            else:
                self.joint_voltages[j] = np.random.uniform(*self.voltage_normal_range)

    # ── Physics degradation ──────────────────────────────────────────────

    def _compute_degradation(self, temp, joint_idx=0):
        """Delegates to ThermalModel."""
        return self.thermal_model.compute_degradation(temp, joint_idx)

    def _apply_thermal_physics(self, action):
        """Apply all enabled degradation effects sequentially."""
        a = self.thermal_model.apply_thermal_physics(action, self.joint_temps)
        if self.enable_current:
            a = self.current_model.apply_current_physics(a, self.joint_currents)
        if self.enable_voltage:
            a = self.voltage_model.apply_voltage_physics(a, self.joint_voltages)
        return a

    # ── Observation building ─────────────────────────────────────────────

    def _build_obs(self, raw_obs):
        """
        Build observation based on mode:

        State-only (use_images=False) — 24-dim flat vector:
          [joint_pos(7) | joint_temps(7) | eef_pos(3) | eef_quat(4) | cube_pos(3)]
          cube_pos is included because the policy has no other way to locate the cube.

        Image mode (use_images=True) — Dict:
          "state": 21-dim [joint_pos(7) | joint_temps(7) | eef_pos(3) | eef_quat(4)]
                   cube_pos is DROPPED — the wrist camera provides visual localisation.
          "image": (3, H, W) uint8 from eye-in-hand camera.
        """
        if "robot0_joint_pos" in raw_obs:
            joint_pos = raw_obs["robot0_joint_pos"].astype(np.float32)
        else:
            # robosuite 1.4.x: reconstruct from cos/sin
            joint_pos = np.arctan2(
                raw_obs["robot0_joint_pos_sin"],
                raw_obs["robot0_joint_pos_cos"],
            ).astype(np.float32)
        eef_pos   = raw_obs["robot0_eef_pos"].astype(np.float32)     # (3,)
        eef_quat  = raw_obs["robot0_eef_quat"].astype(np.float32)    # (4,)

        # Build telemetry block: temps always, currents/voltages if enabled
        telemetry_parts = [self.joint_temps]
        if self.enable_current:
            telemetry_parts.append(self.joint_currents)
        if self.enable_voltage:
            telemetry_parts.append(self.joint_voltages)

        if self.use_images:
            state = np.concatenate([
                joint_pos,         # (7) joint angles
                *telemetry_parts,  # (7+) telemetry channels
                eef_pos,           # (3) end-effector position
                eef_quat,          # (4) end-effector orientation
            ])

            img = raw_obs.get(
                self._image_obs_key,
                np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                ),
            )
            # robosuite returns (H, W, 3) — convert to (3, H, W) for SB3
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)
            return {"state": state, "image": img.astype(np.uint8)}
        else:
            cube_pos = raw_obs.get("cube_pos", np.zeros(3)).astype(np.float32)
            state = np.concatenate([
                joint_pos,         # (7) joint angles
                *telemetry_parts,  # (7+) telemetry channels
                eef_pos,           # (3) end-effector position
                eef_quat,          # (4) end-effector orientation
                cube_pos,          # (3) cube XYZ
            ])
            return state

    # ── Gym interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._sample_all_temps()
        if self.enable_current:
            self._sample_all_currents()
            self.current_model.reset()
        if self.enable_voltage:
            self._sample_all_voltages()

        raw_obs = self.env.reset()
        self.steps = 0
        self.episode_count += 1
        self._last_obs = raw_obs
        self._success_triggered = False
        self._milestones_hit = set()
        # Record the cube's resting height (center of cube sitting on table)
        # so lift reward is measured from actual rest, not from the table surface.
        cube_pos = raw_obs.get("cube_pos", np.zeros(3))
        self._cube_rest_height = cube_pos[2]

        obs = self._build_obs(raw_obs)

        info = {
            "joint_temps": self.joint_temps.copy(),
            "degradation": np.array(
                [self._compute_degradation(t) for t in self.joint_temps]
            ),
            "episode": self.episode_count,
        }
        if self.enable_current:
            info["joint_currents"] = self.joint_currents.copy()
        if self.enable_voltage:
            info["joint_voltages"] = self.joint_voltages.copy()
        return obs, info

    def step(self, action):
        self.steps += 1

        # Apply thermal degradation
        degraded_action = self._apply_thermal_physics(action)

        # Step robosuite
        raw_obs, reward, done, info = self.env.step(degraded_action)
        self._last_obs = raw_obs

        # Check success
        if hasattr(self.env, "_check_success"):
            task_success = bool(self.env._check_success())
        elif hasattr(self.env, "is_success"):
            success = self.env.is_success()
            task_success = (
                success.get("task", False)
                if isinstance(success, dict)
                else bool(success)
            )
        else:
            task_success = False

        obs = self._build_obs(raw_obs)

        # ── Reward ────────────────────────────────────────────────────────
        # Base: robosuite shaped reward (reaching + grasping components)
        shaped_reward = reward * self.reward_scale

        # Cube height above its RESTING position (not the table surface).
        # The cube center sits ~3.1 cm above the table at rest, so we
        # measure lift relative to that baseline to avoid free bonuses.
        cube_pos = raw_obs.get("cube_pos", np.zeros(3))
        rest_h = self._cube_rest_height if self._cube_rest_height else (self._table_height + 0.031)
        lift_delta = max(0.0, cube_pos[2] - rest_h)
        # Also keep absolute height above table for info dict
        height_above_table = max(0.0, cube_pos[2] - self._table_height)

        # Check if robot is currently grasping the cube
        try:
            is_grasping = self.env._check_grasp(
                gripper=self.env.robots[0].gripper,
                object_geoms=self.env.cube,
            )
        except Exception:
            is_grasping = False

        # ① Continuous lift-height reward (when grasping and cube is rising)
        #    Higher lift = more reward — strong gradient all the way up.
        #    This matches the successful outputs_lift2 runs (25K+ rew, 88% SR).
        if is_grasping and lift_delta > 0.001 and self.lift_bonus > 0:
            shaped_reward += lift_delta * self.lift_bonus

        # ② One-time height milestone bonuses (between rest and success)
        #    Resting height ≈ 3.1cm above table, success ≈ 4.0cm above table.
        #    Milestones at +0.3cm, +0.5cm, +0.7cm above rest.
        _MILESTONES = {0.003: 2.0, 0.005: 3.0, 0.007: 5.0}
        for delta_thresh, bonus in _MILESTONES.items():
            if lift_delta >= delta_thresh and delta_thresh not in self._milestones_hit:
                self._milestones_hit.add(delta_thresh)
                shaped_reward += bonus

        # ③ One-time success bonus (only awarded once per episode)
        if task_success and not self._success_triggered:
            shaped_reward += self.success_bonus
            self._success_triggered = True

        # ── Termination ───────────────────────────────────────────────────
        # By default, do NOT terminate on success. This prevents the policy
        # from learning to stall at grasp (which yields more cumulative
        # reward than early success + termination).
        if self.terminate_on_success:
            terminated = bool(task_success)
        else:
            terminated = False
        truncated = self.steps >= self.horizon

        degradations = np.array(
            [self._compute_degradation(t) for t in self.joint_temps]
        )

        info.update({
            "joint_temps": self.joint_temps.copy(),
            "degradation": degradations,
            "avg_degradation": degradations.mean(),
            "n_degraded_joints": int((degradations < 0.8).sum()),
            "task_success": task_success,
            "cube_height_above_table": height_above_table,
            "cube_lift_delta": lift_delta,
            "is_grasping": is_grasping,
            "raw_reward": reward,
        })
        if self.enable_current:
            info["joint_currents"] = self.joint_currents.copy()
        if self.enable_voltage:
            info["joint_voltages"] = self.joint_voltages.copy()

        return obs, shaped_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # Use sim.render for offscreen frame capture (MuJoCo renders upside-down)
            frame = self.env.sim.render(
                camera_name="agentview", height=480, width=640,
            )
            return frame[::-1].copy() if frame is not None else None
        elif self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()

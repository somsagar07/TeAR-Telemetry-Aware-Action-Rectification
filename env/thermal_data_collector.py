"""
Data Collection for Thermal Degradation Curve Discovery.

Two collectors:
  1. SimDataCollector  — sweep temperatures in simulation, measure commanded vs achieved
  2. RealDataCollector — log data during real robot operation (monotonic warm-up)
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class SimDataCollector:
    """
    Collect thermal degradation data in simulation.

    Strategy: For each temperature point, apply a known degradation model (or
    black-box env) and measure how much the action is attenuated. This produces
    (temp, commanded, achieved, torque_ratio) tuples.

    Two modes:
      1. Probe a black-box env that has thermal effects baked in
      2. Probe the degradation function directly (for validation)
    """

    def collect_from_degradation_fn(
        self,
        degradation_fn,
        temp_range: Tuple[float, float] = (20.0, 80.0),
        n_temp_points: int = 50,
        n_samples_per_temp: int = 20,
        noise_scale: float = 0.15,
        n_joints: int = 7,
    ) -> Dict[str, np.ndarray]:
        """
        Collect data by probing a degradation function directly.

        This simulates what you'd observe: command an action, get a degraded
        result, and record the ratio. Noise is added to simulate real conditions.

        Parameters
        ----------
        degradation_fn : callable
            temp → degradation_factor (like _compute_degradation)
        temp_range : (low, high) temperature range to sweep
        n_temp_points : int
            Number of temperature values to sample
        n_samples_per_temp : int
            Number of action samples per temperature point
        noise_scale : float
            Torque noise scale (matching env setting)
        n_joints : int
            Number of joints

        Returns
        -------
        dict with keys: temps, joint_indices, commanded, achieved, torque_ratios, noise_observed
        """
        temps_list = []
        joint_indices_list = []
        commanded_list = []
        achieved_list = []
        ratios_list = []
        noise_list = []

        temp_values = np.linspace(temp_range[0], temp_range[1], n_temp_points)

        for temp in temp_values:
            deg = degradation_fn(temp)
            for joint_idx in range(n_joints):
                for _ in range(n_samples_per_temp):
                    cmd = np.random.uniform(-1.0, 1.0)
                    noise_std = (1.0 - deg) * noise_scale
                    noise = np.random.normal(0, noise_std) if noise_std > 0 else 0.0
                    achieved = np.clip(cmd * deg + noise, -1.0, 1.0)

                    ratio = achieved / cmd if abs(cmd) > 0.01 else deg

                    temps_list.append(temp)
                    joint_indices_list.append(joint_idx)
                    commanded_list.append(cmd)
                    achieved_list.append(achieved)
                    ratios_list.append(deg)  # true degradation (for validation)
                    noise_list.append(abs(achieved - cmd * deg))

        return {
            "temps": np.array(temps_list, dtype=np.float32),
            "joint_indices": np.array(joint_indices_list, dtype=np.int32),
            "commanded": np.array(commanded_list, dtype=np.float32),
            "achieved": np.array(achieved_list, dtype=np.float32),
            "torque_ratios": np.array(ratios_list, dtype=np.float32),
            "noise_observed": np.array(noise_list, dtype=np.float32),
        }

    def collect_from_env(
        self,
        env,
        temp_range: Tuple[float, float] = (20.0, 80.0),
        n_temp_points: int = 30,
        n_action_samples: int = 10,
        action_dim: int = 7,
    ) -> Dict[str, np.ndarray]:
        """
        Collect data by probing an environment with thermal effects.

        Forces all joints to the same temperature, commands known actions,
        observes the effect through env.step().

        Parameters
        ----------
        env : gym.Env
            Must have joint_temps attribute and _apply_thermal_physics/_apply_thermal method.
        temp_range : (low, high)
        n_temp_points : int
        n_action_samples : int
            Number of random actions per temperature point
        action_dim : int

        Returns
        -------
        dict with keys: temps, commanded_actions, degraded_actions
        """
        temps_list = []
        commanded_list = []
        degraded_list = []

        temp_values = np.linspace(temp_range[0], temp_range[1], n_temp_points)

        # Determine which apply method exists
        apply_fn = None
        if hasattr(env, "_apply_thermal_physics"):
            apply_fn = env._apply_thermal_physics
        elif hasattr(env, "_apply_thermal"):
            apply_fn = env._apply_thermal
        else:
            raise AttributeError("Env must have _apply_thermal_physics or _apply_thermal method")

        for temp in temp_values:
            # Force all joints to this temperature
            env.joint_temps = np.full(env.n_joints, temp, dtype=np.float32)

            for _ in range(n_action_samples):
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
                degraded = apply_fn(action)

                temps_list.append(temp)
                commanded_list.append(action.copy())
                degraded_list.append(degraded.copy())

        return {
            "temps": np.array(temps_list, dtype=np.float32),
            "commanded_actions": np.array(commanded_list, dtype=np.float32),
            "degraded_actions": np.array(degraded_list, dtype=np.float32),
        }

    def extract_torque_ratios(
        self,
        env_data: Dict[str, np.ndarray],
        action_indices: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convert env probe data to (temp, torque_ratio) pairs for fitting.

        Parameters
        ----------
        env_data : dict from collect_from_env
        action_indices : which action dimensions to analyze (default: [0,1,2] for position)

        Returns
        -------
        dict with: temps, torque_ratios, joint_indices
        """
        if action_indices is None:
            action_indices = [0, 1, 2]  # position actions

        temps = env_data["temps"]
        commanded = env_data["commanded_actions"]
        degraded = env_data["degraded_actions"]

        out_temps = []
        out_ratios = []
        out_joints = []

        for i in range(len(temps)):
            for idx in action_indices:
                cmd = commanded[i, idx]
                if abs(cmd) > 0.05:  # skip near-zero commands (noisy ratio)
                    ratio = degraded[i, idx] / cmd
                    ratio = np.clip(ratio, 0.0, 1.5)  # clip outliers
                    out_temps.append(temps[i])
                    out_ratios.append(ratio)
                    out_joints.append(0 if idx < 3 else 3)  # map to joint group

        return {
            "temps": np.array(out_temps, dtype=np.float32),
            "torque_ratios": np.array(out_ratios, dtype=np.float32),
            "joint_indices": np.array(out_joints, dtype=np.int32),
        }

    @staticmethod
    def save_data(data: Dict[str, np.ndarray], path: str):
        """Save collected data to .npz file."""
        np.savez(path, **data)

    @staticmethod
    def load_data(path: str) -> Dict[str, np.ndarray]:
        """Load collected data from .npz file."""
        loaded = np.load(path)
        return {k: loaded[k] for k in loaded.files}


class RealDataCollector:
    """
    Collect thermal degradation data from real robot operation.

    Wraps an environment to log (timestamp, joint_temps, commanded, achieved)
    during normal operation. Since temperature only increases monotonically
    in a session, multiple cold-start sessions are needed for full coverage.
    """

    def __init__(self, log_dir: str = "thermal_data"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._session_data = []

    def log_step(
        self,
        timestamp: float,
        joint_temps: np.ndarray,
        commanded_action: np.ndarray,
        achieved_action: np.ndarray,
    ):
        """Log a single step's data."""
        self._session_data.append({
            "timestamp": timestamp,
            "joint_temps": joint_temps.copy(),
            "commanded": commanded_action.copy(),
            "achieved": achieved_action.copy(),
        })

    def save_session(self, session_name: Optional[str] = None):
        """Save current session data to disk."""
        if not self._session_data:
            return

        if session_name is None:
            import time
            session_name = f"session_{int(time.time())}"

        path = os.path.join(self.log_dir, f"{session_name}.npz")
        n = len(self._session_data)
        np.savez(
            path,
            timestamps=np.array([d["timestamp"] for d in self._session_data]),
            joint_temps=np.array([d["joint_temps"] for d in self._session_data]),
            commanded=np.array([d["commanded"] for d in self._session_data]),
            achieved=np.array([d["achieved"] for d in self._session_data]),
        )
        self._session_data = []
        return path

    def merge_sessions(self, session_paths: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Merge multiple session files into a single dataset.

        Parameters
        ----------
        session_paths : list of .npz file paths
            If None, loads all .npz files from log_dir.

        Returns
        -------
        dict with: timestamps, joint_temps, commanded, achieved
        """
        if session_paths is None:
            session_paths = sorted([
                os.path.join(self.log_dir, f)
                for f in os.listdir(self.log_dir)
                if f.endswith(".npz")
            ])

        all_timestamps = []
        all_temps = []
        all_commanded = []
        all_achieved = []

        for path in session_paths:
            data = np.load(path)
            all_timestamps.append(data["timestamps"])
            all_temps.append(data["joint_temps"])
            all_commanded.append(data["commanded"])
            all_achieved.append(data["achieved"])

        return {
            "timestamps": np.concatenate(all_timestamps),
            "joint_temps": np.concatenate(all_temps),
            "commanded": np.concatenate(all_commanded),
            "achieved": np.concatenate(all_achieved),
        }

    def extract_torque_ratios(
        self,
        merged_data: Dict[str, np.ndarray],
        joint_idx: int = 0,
        action_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Extract (temp, torque_ratio) pairs for a specific joint from merged data.

        Parameters
        ----------
        merged_data : from merge_sessions
        joint_idx : which joint's temperature to use
        action_idx : which action dimension to analyze

        Returns
        -------
        dict with: temps, torque_ratios
        """
        temps = merged_data["joint_temps"][:, joint_idx]
        commanded = merged_data["commanded"][:, action_idx]
        achieved = merged_data["achieved"][:, action_idx]

        # Filter out near-zero commands
        mask = np.abs(commanded) > 0.05
        temps = temps[mask]
        ratios = achieved[mask] / commanded[mask]
        ratios = np.clip(ratios, 0.0, 1.5)

        return {
            "temps": temps.astype(np.float32),
            "torque_ratios": ratios.astype(np.float32),
        }

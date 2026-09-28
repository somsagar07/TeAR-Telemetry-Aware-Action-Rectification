"""
Configurable Thermal Degradation Model.

Supports three modes:
  1. Predefined curves  — linear, exponential, sigmoid, polynomial
  2. Custom callables    — user-defined temp→degradation functions
  3. Auto-discovered     — fitted from collected (temp, torque_ratio) data

Granularity options:
  - "single"    : one curve shared across all joints
  - "per_group" : separate curves for position (joints 0-3) and rotation (joints 3-6)
  - "per_joint" : independent curve per joint (7 total for Panda)
"""

import json
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


# ═════════════════════════════════════════════════════════════════════════════
#  Predefined Curve Functions
# ═════════════════════════════════════════════════════════════════════════════

def linear_degradation(temp, onset=43.0, full=75.0, min_cap=0.05):
    """Linear ramp from 1.0 at onset to min_cap at full. (Current default.)"""
    if temp <= onset:
        return 1.0
    elif temp >= full:
        return min_cap
    t = (temp - onset) / (full - onset)
    return 1.0 - (1.0 - min_cap) * t


def exponential_degradation(temp, onset=43.0, k=0.08, min_cap=0.05):
    """Exponential decay: 1.0 * exp(-k * (temp - onset)), clamped to [min_cap, 1]."""
    if temp <= onset:
        return 1.0
    return max(min_cap, np.exp(-k * (temp - onset)))


def sigmoid_degradation(temp, midpoint=59.0, k=0.2, min_cap=0.05):
    """Sigmoid (logistic) curve centered at midpoint."""
    raw = 1.0 / (1.0 + np.exp(k * (temp - midpoint)))
    return max(min_cap, raw)


def polynomial_degradation(temp, onset=43.0, full=75.0, degree=2, min_cap=0.05):
    """Polynomial ramp of given degree between onset and full."""
    if temp <= onset:
        return 1.0
    elif temp >= full:
        return min_cap
    t = (temp - onset) / (full - onset)
    return 1.0 - (1.0 - min_cap) * (t ** degree)


def kinky_degradation(temp, breakpoints=None, values=None, min_cap=0.05):
    """Piecewise-linear capacity with sharp kinks (e.g. motor protection thresholds).
    Polynomial regression has Runge-style oscillation around kinks; an MLP / Transformer
    can fit them sharply. Used for the curve-mismatch supplementary experiment.

    Defaults: 4-segment curve with kinks at T=45, 55, 65 deg C. Mimics a motor that
    derates in three discrete stages (lubricant viscosity threshold; coil resistance
    knee; protection circuit cutback).
    """
    if breakpoints is None:
        breakpoints = [20.0, 45.0, 55.0, 65.0, 80.0]
        values      = [1.00, 1.00, 0.70, 0.30, 0.05]
    if temp <= breakpoints[0]:
        return max(values[0], min_cap)
    if temp >= breakpoints[-1]:
        return max(values[-1], min_cap)
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= temp <= breakpoints[i + 1]:
            t = (temp - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            v = values[i] + t * (values[i + 1] - values[i])
            return max(v, min_cap)
    return min_cap


# Registry mapping names to (function, default_params)
def sin_degradation(temp, onset=43.0, full=75.0, min_cap=0.05):
    """Non-monotonic oscillating capacity (worst-case curve-family mismatch).
    Capacity is 1 in the nominal regime; in the stressed band it oscillates
    smoothly between min_cap and 1 — neither monotonically degrading nor
    factorisable by severity tier."""
    if temp <= onset:
        return 1.0
    if temp >= full:
        return min_cap
    u = (temp - onset) / (full - onset)
    osc = 0.5 * (1.0 + np.cos(3.0 * np.pi * u))
    return max(min_cap, min_cap + (1.0 - min_cap) * osc)


CURVE_REGISTRY = {
    "linear": (linear_degradation, {"onset": 43.0, "full": 75.0, "min_cap": 0.05}),
    "exponential": (exponential_degradation, {"onset": 43.0, "k": 0.08, "min_cap": 0.05}),
    "sigmoid": (sigmoid_degradation, {"midpoint": 59.0, "k": 0.2, "min_cap": 0.05}),
    "polynomial": (polynomial_degradation, {"onset": 43.0, "full": 75.0, "degree": 2, "min_cap": 0.05}),
    "kinky": (kinky_degradation, {"min_cap": 0.05}),
    "sin": (sin_degradation, {"onset": 43.0, "full": 75.0, "min_cap": 0.05}),
}


def get_curve_function(name: str, params: Optional[dict] = None) -> Callable:
    """Return a callable temp→degradation for a registered curve type."""
    if name not in CURVE_REGISTRY:
        raise ValueError(f"Unknown curve type '{name}'. Available: {list(CURVE_REGISTRY.keys())}")
    func, defaults = CURVE_REGISTRY[name]
    merged = {**defaults, **(params or {})}
    return lambda temp: func(temp, **merged)


# ═════════════════════════════════════════════════════════════════════════════
#  Thermal Model
# ═════════════════════════════════════════════════════════════════════════════

class ThermalModel:
    """
    Configurable thermal degradation model.

    Holds per-joint (or per-group / single) degradation and noise curves.
    Replaces hardcoded degradation logic in ThermalLiftEnv / ThermalWrapperEnv.

    Parameters
    ----------
    n_joints : int
        Number of robot joints (7 for Panda).
    granularity : str
        "single", "per_group", or "per_joint".
    torque_noise_scale : float
        Base noise multiplier (noise_std = (1 - capability) * torque_noise_scale).
    position_joints : list
        Joint indices driving position actions (default [0,1,2,3]).
    rotation_joints : list
        Joint indices driving rotation actions (default [3,4,5,6]).
    """

    def __init__(
        self,
        n_joints: int = 7,
        granularity: str = "single",
        torque_noise_scale: float = 0.15,
        position_joints: Optional[List[int]] = None,
        rotation_joints: Optional[List[int]] = None,
    ):
        self.n_joints = n_joints
        self.granularity = granularity
        self.torque_noise_scale = torque_noise_scale
        self.position_joints = position_joints or [0, 1, 2, 3]
        self.rotation_joints = rotation_joints or [3, 4, 5, 6]

        # Degradation curves: mapping key → callable(temp) → float in [0, 1]
        # Keys depend on granularity:
        #   "single"    → {"all": fn}
        #   "per_group" → {"position": fn, "rotation": fn}
        #   "per_joint" → {"joint_0": fn, ..., "joint_6": fn}
        self._curves: Dict[str, Callable] = {}

        # Noise curves: same keys → callable(temp) → noise_std
        # If None, noise is derived from degradation: (1 - deg) * torque_noise_scale
        self._noise_curves: Dict[str, Optional[Callable]] = {}

        # Metadata for serialization
        self._curve_specs: Dict[str, dict] = {}
        self._mode: str = "uninitialized"

    # ── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def from_predefined(
        cls,
        curve_type: Union[str, Dict[str, Tuple[str, dict]]],
        granularity: str = "single",
        params: Optional[dict] = None,
        torque_noise_scale: float = 0.15,
        n_joints: int = 7,
    ) -> "ThermalModel":
        """
        Create model from predefined curve type(s).

        Parameters
        ----------
        curve_type : str or dict
            If str: same curve for all keys (e.g., "linear").
            If dict: mapping group/joint keys to (curve_name, params) tuples.
              e.g. {"position": ("exponential", {"k": 0.1}),
                     "rotation": ("linear", {})}
        granularity : str
            "single", "per_group", or "per_joint".
        params : dict, optional
            Parameters for the curve (only when curve_type is str).
        """
        model = cls(n_joints=n_joints, granularity=granularity,
                     torque_noise_scale=torque_noise_scale)
        model._mode = "predefined"

        if isinstance(curve_type, str):
            keys = model._get_keys()
            for key in keys:
                model._curves[key] = get_curve_function(curve_type, params)
                model._curve_specs[key] = {"type": curve_type, "params": params or {}}
        elif isinstance(curve_type, dict):
            for key, (ctype, cparams) in curve_type.items():
                model._curves[key] = get_curve_function(ctype, cparams)
                model._curve_specs[key] = {"type": ctype, "params": cparams}
        else:
            raise TypeError("curve_type must be str or dict")

        return model

    @classmethod
    def from_callable(
        cls,
        fn: Union[Callable, Dict[str, Callable]],
        granularity: str = "single",
        torque_noise_scale: float = 0.15,
        noise_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
        n_joints: int = 7,
    ) -> "ThermalModel":
        """
        Create model from user-provided callable(s).

        Parameters
        ----------
        fn : callable or dict
            If callable: temp→degradation, used for all keys.
            If dict: mapping keys to callables.
        noise_fn : callable or dict, optional
            If provided, temp→noise_std. Otherwise derived from degradation.
        """
        model = cls(n_joints=n_joints, granularity=granularity,
                     torque_noise_scale=torque_noise_scale)
        model._mode = "custom"

        keys = model._get_keys()
        if callable(fn):
            for key in keys:
                model._curves[key] = fn
                model._curve_specs[key] = {"type": "custom", "params": {}}
        elif isinstance(fn, dict):
            for key, f in fn.items():
                model._curves[key] = f
                model._curve_specs[key] = {"type": "custom", "params": {}}
        else:
            raise TypeError("fn must be callable or dict of callables")

        if noise_fn is not None:
            if callable(noise_fn):
                for key in keys:
                    model._noise_curves[key] = noise_fn
            elif isinstance(noise_fn, dict):
                model._noise_curves.update(noise_fn)

        return model

    @classmethod
    def from_data(
        cls,
        temps: np.ndarray,
        torque_ratios: np.ndarray,
        noise_stds: Optional[np.ndarray] = None,
        joint_indices: Optional[np.ndarray] = None,
        granularity: str = "single",
        curve_type: Optional[str] = None,
        torque_noise_scale: float = 0.15,
        n_joints: int = 7,
    ) -> Tuple["ThermalModel", "FitReport"]:
        """
        Create model by fitting curves to collected data.

        Imports ThermalCurveFitter internally to avoid circular deps.

        Parameters
        ----------
        temps : (N,) array of temperatures
        torque_ratios : (N,) array of observed torque ratios (commanded/achieved)
        noise_stds : (N,) optional array of observed noise standard deviations
        joint_indices : (N,) optional array of joint indices for per-joint fitting
        granularity : str
        curve_type : str or None
            If None, auto-select best fit. If str, fit only that type.

        Returns
        -------
        (ThermalModel, FitReport)
        """
        from .thermal_curve_fitter import ThermalCurveFitter

        fitter = ThermalCurveFitter()
        model, report = fitter.fit(
            temps=temps,
            torque_ratios=torque_ratios,
            noise_stds=noise_stds,
            joint_indices=joint_indices,
            granularity=granularity,
            curve_type=curve_type,
            torque_noise_scale=torque_noise_scale,
            n_joints=n_joints,
        )
        return model, report

    # ── Core methods ─────────────────────────────────────────────────────

    def compute_degradation(self, temp: float, joint_idx: int = 0) -> float:
        """Compute degradation factor for a single joint at given temperature."""
        key = self._joint_to_key(joint_idx)
        return float(self._curves[key](temp))

    def compute_noise_std(self, temp: float, joint_idx: int = 0) -> float:
        """Compute noise standard deviation for a joint at given temperature."""
        key = self._joint_to_key(joint_idx)
        if key in self._noise_curves and self._noise_curves[key] is not None:
            return float(self._noise_curves[key](temp))
        deg = self.compute_degradation(temp, joint_idx)
        return (1.0 - deg) * self.torque_noise_scale

    def apply_thermal_physics(self, action: np.ndarray, joint_temps: np.ndarray) -> np.ndarray:
        """
        Apply temperature-based degradation to robot actions.

        Position (0:3) → driven by position_joints
        Rotation (3:6) → driven by rotation_joints
        Gripper  (6)   → unchanged
        """
        degraded = action.copy()
        degradations = np.array([
            self.compute_degradation(t, j)
            for j, t in enumerate(joint_temps)
        ])

        # Position actions
        pos_cap = degradations[self.position_joints].mean()
        for i in range(3):
            degraded[i] *= pos_cap
            if pos_cap < 1.0:
                noise = self._get_group_noise(joint_temps, self.position_joints)
                degraded[i] += np.random.normal(0, noise)

        # Rotation actions
        rot_cap = degradations[self.rotation_joints].mean()
        for i in range(3, 6):
            degraded[i] *= rot_cap
            if rot_cap < 1.0:
                noise = self._get_group_noise(joint_temps, self.rotation_joints)
                degraded[i] += np.random.normal(0, noise)

        return np.clip(degraded, -1.0, 1.0)

    def _get_group_noise(self, joint_temps: np.ndarray, joint_indices: List[int]) -> float:
        """Average noise across a group of joints."""
        noises = [self.compute_noise_std(joint_temps[j], j) for j in joint_indices]
        return float(np.mean(noises))

    # ── Key management ───────────────────────────────────────────────────

    def _get_keys(self) -> List[str]:
        if self.granularity == "single":
            return ["all"]
        elif self.granularity == "per_group":
            return ["position", "rotation"]
        elif self.granularity == "per_joint":
            return [f"joint_{i}" for i in range(self.n_joints)]
        raise ValueError(f"Unknown granularity: {self.granularity}")

    def _joint_to_key(self, joint_idx: int) -> str:
        if self.granularity == "single":
            return "all"
        elif self.granularity == "per_group":
            return "position" if joint_idx in self.position_joints else "rotation"
        elif self.granularity == "per_joint":
            return f"joint_{joint_idx}"
        raise ValueError(f"Unknown granularity: {self.granularity}")

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model config to JSON (only works for predefined/fitted curves)."""
        if self._mode == "custom":
            raise ValueError("Cannot serialize custom callables. Use predefined or fitted curves.")

        data = {
            "mode": self._mode,
            "n_joints": self.n_joints,
            "granularity": self.granularity,
            "torque_noise_scale": self.torque_noise_scale,
            "position_joints": self.position_joints,
            "rotation_joints": self.rotation_joints,
            "curves": self._curve_specs,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ThermalModel":
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)

        model = cls(
            n_joints=data["n_joints"],
            granularity=data["granularity"],
            torque_noise_scale=data["torque_noise_scale"],
            position_joints=data.get("position_joints", [0, 1, 2, 3]),
            rotation_joints=data.get("rotation_joints", [3, 4, 5, 6]),
        )
        model._mode = data["mode"]
        model._curve_specs = data["curves"]

        for key, spec in data["curves"].items():
            model._curves[key] = get_curve_function(spec["type"], spec["params"])

        return model

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"ThermalModel(mode={self._mode!r}, granularity={self.granularity!r}, "
            f"noise_scale={self.torque_noise_scale}, curves={list(self._curves.keys())})"
        )

    def summary(self) -> str:
        """Human-readable summary of the model configuration."""
        lines = [
            f"ThermalModel — {self._mode} mode, {self.granularity} granularity",
            f"  Joints: {self.n_joints}, Noise scale: {self.torque_noise_scale}",
            f"  Position joints: {self.position_joints}",
            f"  Rotation joints: {self.rotation_joints}",
            "  Curves:",
        ]
        for key, spec in self._curve_specs.items():
            lines.append(f"    {key}: {spec['type']} {spec.get('params', {})}")

        # Sample degradation at key temperatures
        lines.append("  Sample degradation (joint 0):")
        for temp in [20, 30, 43, 50, 60, 70, 75, 80]:
            deg = self.compute_degradation(float(temp), 0)
            lines.append(f"    {temp}C -> {deg:.3f}")

        return "\n".join(lines)

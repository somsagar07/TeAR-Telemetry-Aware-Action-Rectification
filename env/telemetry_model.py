"""
Telemetry-Aware Degradation Models: Current & Voltage.

Extends the thermal degradation framework with two additional telemetry
channels, each with its own physics-based degradation effect:

  Temperature (existing) → torque capacity reduction + noise
  Current (new)          → torque saturation / ripple at high draw
  Voltage (new)          → velocity limiting at low supply voltage

Each model uses the same curve infrastructure (linear, exponential,
sigmoid, polynomial) but with domain-appropriate default parameters.

Combined via TelemetryModel which orchestrates all three.
"""

import json
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from .thermal_model import ThermalModel


# ═════════════════════════════════════════════════════════════════════════════
#  Current Degradation Curves
#  Input: current_ratio (fraction of rated current, 0.0 – 1.2+)
#  Output: degradation factor in [0, 1]
#  Physics: T = Kt * I.  At high current Kt saturates, torque ripple grows.
# ═════════════════════════════════════════════════════════════════════════════

def linear_current_degradation(current, onset=0.60, full=1.0, min_cap=0.10):
    """Linear ramp from 1.0 at onset to min_cap at full."""
    if current <= onset:
        return 1.0
    elif current >= full:
        return min_cap
    t = (current - onset) / (full - onset)
    return 1.0 - (1.0 - min_cap) * t


def exponential_current_degradation(current, onset=0.60, k=5.0, min_cap=0.10):
    """Exponential decay past onset: exp(-k * (current - onset))."""
    if current <= onset:
        return 1.0
    return max(min_cap, np.exp(-k * (current - onset)))


def sigmoid_current_degradation(current, midpoint=0.80, k=12.0, min_cap=0.10):
    """Sigmoid drop centered at midpoint."""
    raw = 1.0 / (1.0 + np.exp(k * (current - midpoint)))
    return max(min_cap, raw)


def polynomial_current_degradation(current, onset=0.60, full=1.0, degree=2, min_cap=0.10):
    """Polynomial ramp between onset and full."""
    if current <= onset:
        return 1.0
    elif current >= full:
        return min_cap
    t = (current - onset) / (full - onset)
    return 1.0 - (1.0 - min_cap) * (t ** degree)


def kinky_current_degradation(current, breakpoints=None, values=None, min_cap=0.10):
    """Piecewise-linear current capacity with sharp kinks (motor protection thresholds)."""
    if breakpoints is None:
        breakpoints = [0.00, 0.55, 0.70, 0.85, 1.10]
        values      = [1.00, 1.00, 0.65, 0.30, 0.10]
    if current <= breakpoints[0]: return max(values[0], min_cap)
    if current >= breakpoints[-1]: return max(values[-1], min_cap)
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= current <= breakpoints[i + 1]:
            t = (current - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            return max(values[i] + t * (values[i + 1] - values[i]), min_cap)
    return min_cap


CURRENT_CURVE_REGISTRY = {
    "linear":      (linear_current_degradation,      {"onset": 0.60, "full": 1.0, "min_cap": 0.10}),
    "exponential": (exponential_current_degradation,  {"onset": 0.60, "k": 5.0, "min_cap": 0.10}),
    "sigmoid":     (sigmoid_current_degradation,      {"midpoint": 0.80, "k": 12.0, "min_cap": 0.10}),
    "polynomial":  (polynomial_current_degradation,   {"onset": 0.60, "full": 1.0, "degree": 2, "min_cap": 0.10}),
    "kinky":       (kinky_current_degradation,        {"min_cap": 0.10}),
}


# ═════════════════════════════════════════════════════════════════════════════
#  Voltage Degradation Curves
#  Input: voltage_ratio (fraction of rated voltage, 0.0 – 1.1)
#  Output: degradation factor in [0, 1]
#  Physics: V_BEMF = Ke * ω.  Low voltage caps max speed → velocity limiting.
#  NOTE: direction is INVERTED vs temperature — LOW voltage = HIGH degradation.
# ═════════════════════════════════════════════════════════════════════════════

def linear_voltage_degradation(voltage, onset=0.90, full=0.50, min_cap=0.10):
    """Linear ramp: full capacity above onset, min_cap at or below full.
    onset > full because degradation starts when voltage DROPS below onset."""
    if voltage >= onset:
        return 1.0
    elif voltage <= full:
        return min_cap
    t = (onset - voltage) / (onset - full)
    return 1.0 - (1.0 - min_cap) * t


def exponential_voltage_degradation(voltage, onset=0.90, k=5.0, min_cap=0.10):
    """Exponential decay as voltage drops below onset."""
    if voltage >= onset:
        return 1.0
    return max(min_cap, np.exp(-k * (onset - voltage)))


def sigmoid_voltage_degradation(voltage, midpoint=0.70, k=12.0, min_cap=0.10):
    """Sigmoid curve centered at midpoint. Higher voltage = higher capacity."""
    raw = 1.0 / (1.0 + np.exp(k * (midpoint - voltage)))
    return max(min_cap, raw)


def polynomial_voltage_degradation(voltage, onset=0.90, full=0.50, degree=2, min_cap=0.10):
    """Polynomial ramp as voltage drops from onset to full."""
    if voltage >= onset:
        return 1.0
    elif voltage <= full:
        return min_cap
    t = (onset - voltage) / (onset - full)
    return 1.0 - (1.0 - min_cap) * (t ** degree)


def kinky_voltage_degradation(voltage, breakpoints=None, values=None, min_cap=0.10):
    """Piecewise-linear voltage capacity with sharp kinks (brownout / cutback stages).
    Direction inverted: low voltage = high degradation."""
    if breakpoints is None:
        breakpoints = [0.40, 0.55, 0.70, 0.85, 1.10]  # ascending
        values      = [0.10, 0.30, 0.60, 1.00, 1.00]  # at breakpoint[i], capacity = values[i]
    if voltage <= breakpoints[0]: return max(values[0], min_cap)
    if voltage >= breakpoints[-1]: return max(values[-1], min_cap)
    for i in range(len(breakpoints) - 1):
        if breakpoints[i] <= voltage <= breakpoints[i + 1]:
            t = (voltage - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            return max(values[i] + t * (values[i + 1] - values[i]), min_cap)
    return min_cap


VOLTAGE_CURVE_REGISTRY = {
    "linear":      (linear_voltage_degradation,      {"onset": 0.90, "full": 0.50, "min_cap": 0.10}),
    "exponential": (exponential_voltage_degradation,  {"onset": 0.90, "k": 5.0, "min_cap": 0.10}),
    "sigmoid":     (sigmoid_voltage_degradation,      {"midpoint": 0.70, "k": 12.0, "min_cap": 0.10}),
    "polynomial":  (polynomial_voltage_degradation,   {"onset": 0.90, "full": 0.50, "degree": 2, "min_cap": 0.10}),
    "kinky":       (kinky_voltage_degradation,        {"min_cap": 0.10}),
}


# ═════════════════════════════════════════════════════════════════════════════
#  Helper: get curve function from a registry
# ═════════════════════════════════════════════════════════════════════════════

def _get_curve(registry, name, params=None):
    if name not in registry:
        raise ValueError(f"Unknown curve '{name}'. Available: {list(registry.keys())}")
    func, defaults = registry[name]
    merged = {**defaults, **(params or {})}
    return lambda val: func(val, **merged)


# ═════════════════════════════════════════════════════════════════════════════
#  CurrentModel
# ═════════════════════════════════════════════════════════════════════════════

class CurrentModel:
    """
    Models the effect of motor current draw on joint performance.

    Physics: T = Kt * I.  At high current (near stall), Kt saturates,
    producing torque ripple and reduced effective torque.

    Degradation effect on actions:
      - Torque scaling (like temperature)
      - Additional sinusoidal ripple noise (motor cogging / torque ripple)

    Parameters
    ----------
    n_joints : int
    ripple_scale : float
        Amplitude of torque ripple as fraction of (1 - degradation).
    noise_scale : float
        Gaussian noise scale (same semantics as thermal noise).
    """

    def __init__(self, n_joints=7, ripple_scale=0.10, noise_scale=0.12):
        self.n_joints = n_joints
        self.ripple_scale = ripple_scale
        self.noise_scale = noise_scale
        self._curves: Dict[str, Callable] = {}
        self._curve_specs: Dict[str, dict] = {}
        self._step_counter = 0  # for ripple phase

    @classmethod
    def from_predefined(cls, curve_type="linear", params=None,
                        n_joints=7, ripple_scale=0.10, noise_scale=0.12):
        model = cls(n_joints=n_joints, ripple_scale=ripple_scale,
                    noise_scale=noise_scale)
        fn = _get_curve(CURRENT_CURVE_REGISTRY, curve_type, params)
        for j in range(n_joints):
            key = f"joint_{j}"
            model._curves[key] = fn
            model._curve_specs[key] = {"type": curve_type, "params": params or {}}
        return model

    def compute_degradation(self, current_ratio, joint_idx=0):
        key = f"joint_{joint_idx}"
        return float(self._curves[key](current_ratio))

    def apply_current_physics(self, action, joint_currents):
        """
        Apply current-based degradation: torque saturation + ripple.

        Position (0:3) and rotation (3:6) actions are scaled by the
        average degradation of their joint groups. Ripple noise is
        added proportional to how far past the onset the current is.
        Gripper (6) is unaffected.
        """
        degraded = action.copy()
        degs = np.array([
            self.compute_degradation(joint_currents[j], j)
            for j in range(min(len(joint_currents), self.n_joints))
        ])

        self._step_counter += 1

        # Position actions (0:3) — driven by joints 0-3
        pos_cap = degs[:4].mean() if len(degs) >= 4 else degs.mean()
        for i in range(3):
            degraded[i] *= pos_cap
            if pos_cap < 1.0:
                # Gaussian noise (like thermal)
                degraded[i] += np.random.normal(0, (1.0 - pos_cap) * self.noise_scale)
                # Sinusoidal ripple (motor cogging at high current)
                ripple = (1.0 - pos_cap) * self.ripple_scale * np.sin(
                    self._step_counter * 0.5 + i * 2.094)  # 120° phase offset
                degraded[i] += ripple

        # Rotation actions (3:6) — driven by joints 3-6
        rot_cap = degs[3:7].mean() if len(degs) >= 7 else degs.mean()
        for i in range(3, 6):
            degraded[i] *= rot_cap
            if rot_cap < 1.0:
                degraded[i] += np.random.normal(0, (1.0 - rot_cap) * self.noise_scale)
                ripple = (1.0 - rot_cap) * self.ripple_scale * np.sin(
                    self._step_counter * 0.5 + i * 2.094)
                degraded[i] += ripple

        return np.clip(degraded, -1.0, 1.0)

    def reset(self):
        self._step_counter = 0

    def save(self, path):
        data = {"n_joints": self.n_joints, "ripple_scale": self.ripple_scale,
                "noise_scale": self.noise_scale, "curves": self._curve_specs}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        model = cls(n_joints=data["n_joints"],
                    ripple_scale=data.get("ripple_scale", 0.10),
                    noise_scale=data.get("noise_scale", 0.12))
        for key, spec in data["curves"].items():
            model._curves[key] = _get_curve(CURRENT_CURVE_REGISTRY,
                                            spec["type"], spec.get("params"))
            model._curve_specs[key] = spec
        return model

    def summary(self):
        lines = [
            f"CurrentModel — ripple_scale={self.ripple_scale}, noise_scale={self.noise_scale}",
            f"  Joints: {self.n_joints}",
            "  Sample degradation (joint 0):",
        ]
        for cur in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
            deg = self.compute_degradation(cur, 0)
            lines.append(f"    {cur:.1f} I_rated -> {deg:.3f}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  VoltageModel
# ═════════════════════════════════════════════════════════════════════════════

class VoltageModel:
    """
    Models the effect of supply voltage on joint performance.

    Physics: V_BEMF = Ke * ω.  Low voltage caps max angular velocity,
    causing the joint to be slow / unable to track commanded trajectories.

    Degradation effect on actions:
      - Velocity limiting (action magnitude scaling)
      - Position tracking lag (noise representing overshoot/undershoot)

    Parameters
    ----------
    n_joints : int
    lag_scale : float
        Scale of tracking-lag noise as fraction of (1 - degradation).
    """

    def __init__(self, n_joints=7, lag_scale=0.10):
        self.n_joints = n_joints
        self.lag_scale = lag_scale
        self._curves: Dict[str, Callable] = {}
        self._curve_specs: Dict[str, dict] = {}

    @classmethod
    def from_predefined(cls, curve_type="linear", params=None,
                        n_joints=7, lag_scale=0.10):
        model = cls(n_joints=n_joints, lag_scale=lag_scale)
        fn = _get_curve(VOLTAGE_CURVE_REGISTRY, curve_type, params)
        for j in range(n_joints):
            key = f"joint_{j}"
            model._curves[key] = fn
            model._curve_specs[key] = {"type": curve_type, "params": params or {}}
        return model

    def compute_degradation(self, voltage_ratio, joint_idx=0):
        key = f"joint_{joint_idx}"
        return float(self._curves[key](voltage_ratio))

    def apply_voltage_physics(self, action, joint_voltages):
        """
        Apply voltage-based degradation: velocity limiting + tracking lag.

        Low voltage reduces the maximum magnitude of each action dimension
        (joint can't move as fast) and adds tracking-lag noise (overshoot).
        Gripper (6) is unaffected.
        """
        degraded = action.copy()
        degs = np.array([
            self.compute_degradation(joint_voltages[j], j)
            for j in range(min(len(joint_voltages), self.n_joints))
        ])

        # Position actions (0:3)
        pos_cap = degs[:4].mean() if len(degs) >= 4 else degs.mean()
        for i in range(3):
            # Velocity limiting: cap the magnitude
            degraded[i] *= pos_cap
            if pos_cap < 1.0:
                # Tracking lag: random overshoot/undershoot
                degraded[i] += np.random.normal(0, (1.0 - pos_cap) * self.lag_scale)

        # Rotation actions (3:6)
        rot_cap = degs[3:7].mean() if len(degs) >= 7 else degs.mean()
        for i in range(3, 6):
            degraded[i] *= rot_cap
            if rot_cap < 1.0:
                degraded[i] += np.random.normal(0, (1.0 - rot_cap) * self.lag_scale)

        return np.clip(degraded, -1.0, 1.0)

    def save(self, path):
        data = {"n_joints": self.n_joints, "lag_scale": self.lag_scale,
                "curves": self._curve_specs}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        model = cls(n_joints=data["n_joints"],
                    lag_scale=data.get("lag_scale", 0.10))
        for key, spec in data["curves"].items():
            model._curves[key] = _get_curve(VOLTAGE_CURVE_REGISTRY,
                                            spec["type"], spec.get("params"))
            model._curve_specs[key] = spec
        return model

    def summary(self):
        lines = [
            f"VoltageModel — lag_scale={self.lag_scale}",
            f"  Joints: {self.n_joints}",
            "  Sample degradation (joint 0):",
        ]
        for v in [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40]:
            deg = self.compute_degradation(v, 0)
            lines.append(f"    {v:.2f} V_rated -> {deg:.3f}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  TelemetryModel — combines Temperature + Current + Voltage
# ═════════════════════════════════════════════════════════════════════════════

class TelemetryModel:
    """
    Unified telemetry degradation model combining temperature, current,
    and voltage effects.

    Each channel applies its physics independently and the effects compose:
      1. Temperature → torque capacity + gaussian noise
      2. Current → torque saturation + ripple
      3. Voltage → velocity limiting + tracking lag

    The three effects are applied sequentially to the action vector.

    Parameters
    ----------
    thermal_model : ThermalModel or None
    current_model : CurrentModel or None
    voltage_model : VoltageModel or None
    enable_thermal : bool
    enable_current : bool
    enable_voltage : bool
    """

    def __init__(
        self,
        thermal_model: Optional[ThermalModel] = None,
        current_model: Optional[CurrentModel] = None,
        voltage_model: Optional[VoltageModel] = None,
        enable_thermal: bool = True,
        enable_current: bool = True,
        enable_voltage: bool = True,
        n_joints: int = 7,
    ):
        self.n_joints = n_joints
        self.enable_thermal = enable_thermal
        self.enable_current = enable_current
        self.enable_voltage = enable_voltage

        # Create defaults if not provided
        if thermal_model is not None:
            self.thermal = thermal_model
        else:
            self.thermal = ThermalModel.from_predefined(
                "linear", n_joints=n_joints)

        if current_model is not None:
            self.current = current_model
        else:
            self.current = CurrentModel.from_predefined(
                "linear", n_joints=n_joints)

        if voltage_model is not None:
            self.voltage = voltage_model
        else:
            self.voltage = VoltageModel.from_predefined(
                "linear", n_joints=n_joints)

    def apply_degradation(self, action, joint_temps=None,
                          joint_currents=None, joint_voltages=None):
        """
        Apply all enabled degradation effects sequentially.

        Returns
        -------
        degraded_action : np.ndarray, shape (act_dim,)
        """
        a = action.copy()

        if self.enable_thermal and joint_temps is not None:
            a = self.thermal.apply_thermal_physics(a, joint_temps)

        if self.enable_current and joint_currents is not None:
            a = self.current.apply_current_physics(a, joint_currents)

        if self.enable_voltage and joint_voltages is not None:
            a = self.voltage.apply_voltage_physics(a, joint_voltages)

        return np.clip(a, -1.0, 1.0)

    def reset(self):
        """Reset per-episode state (ripple counter, etc.)."""
        if hasattr(self.current, 'reset'):
            self.current.reset()

    def compute_all_degradations(self, joint_temps=None,
                                 joint_currents=None, joint_voltages=None):
        """
        Compute per-joint degradation factors for each channel.

        Returns dict with keys 'thermal', 'current', 'voltage', each
        containing an (n_joints,) array of degradation factors.
        """
        result = {}
        if self.enable_thermal and joint_temps is not None:
            result["thermal"] = np.array([
                self.thermal.compute_degradation(joint_temps[j], j)
                for j in range(self.n_joints)
            ])
        if self.enable_current and joint_currents is not None:
            result["current"] = np.array([
                self.current.compute_degradation(joint_currents[j], j)
                for j in range(self.n_joints)
            ])
        if self.enable_voltage and joint_voltages is not None:
            result["voltage"] = np.array([
                self.voltage.compute_degradation(joint_voltages[j], j)
                for j in range(self.n_joints)
            ])
        return result

    def summary(self):
        parts = [f"TelemetryModel (n_joints={self.n_joints})"]
        if self.enable_thermal:
            parts.append(f"\n[Thermal — ENABLED]\n{self.thermal.summary()}")
        else:
            parts.append("\n[Thermal — DISABLED]")
        if self.enable_current:
            parts.append(f"\n[Current — ENABLED]\n{self.current.summary()}")
        else:
            parts.append("\n[Current — DISABLED]")
        if self.enable_voltage:
            parts.append(f"\n[Voltage — ENABLED]\n{self.voltage.summary()}")
        else:
            parts.append("\n[Voltage — DISABLED]")
        return "\n".join(parts)

    def save(self, path):
        data = {
            "n_joints": self.n_joints,
            "enable_thermal": self.enable_thermal,
            "enable_current": self.enable_current,
            "enable_voltage": self.enable_voltage,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self):
        channels = []
        if self.enable_thermal: channels.append("thermal")
        if self.enable_current: channels.append("current")
        if self.enable_voltage: channels.append("voltage")
        return f"TelemetryModel(channels={channels}, n_joints={self.n_joints})"

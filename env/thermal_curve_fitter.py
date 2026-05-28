"""
Curve Fitting Pipeline for Thermal Degradation Models.

Fits predefined curve types to (temperature, torque_ratio) data.
Supports auto-selection (try all, pick lowest RMSE) or user-specified type.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.optimize import curve_fit

from .thermal_model import (
    CURVE_REGISTRY,
    ThermalModel,
    get_curve_function,
    linear_degradation,
    exponential_degradation,
    sigmoid_degradation,
    polynomial_degradation,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Scipy-compatible curve wrappers (take temp array + params, return array)
# ═════════════════════════════════════════════════════════════════════════════

def _linear_fit(temp, onset, full, min_cap):
    return np.array([linear_degradation(t, onset, full, min_cap) for t in temp])


def _exponential_fit(temp, onset, k, min_cap):
    return np.array([exponential_degradation(t, onset, k, min_cap) for t in temp])


def _sigmoid_fit(temp, midpoint, k, min_cap):
    return np.array([sigmoid_degradation(t, midpoint, k, min_cap) for t in temp])


def _polynomial_fit_deg2(temp, onset, full, min_cap):
    return np.array([polynomial_degradation(t, onset, full, 2, min_cap) for t in temp])


def _polynomial_fit_deg3(temp, onset, full, min_cap):
    return np.array([polynomial_degradation(t, onset, full, 3, min_cap) for t in temp])


# Fitting specs: (scipy_func, initial_guess, bounds_low, bounds_high, param_names, curve_registry_name)
FIT_SPECS = {
    "linear": (
        _linear_fit,
        [43.0, 75.0, 0.05],
        [20.0, 50.0, 0.0],
        [60.0, 100.0, 0.3],
        ["onset", "full", "min_cap"],
        "linear",
    ),
    "exponential": (
        _exponential_fit,
        [43.0, 0.08, 0.05],
        [20.0, 0.001, 0.0],
        [60.0, 1.0, 0.3],
        ["onset", "k", "min_cap"],
        "exponential",
    ),
    "sigmoid": (
        _sigmoid_fit,
        [59.0, 0.2, 0.05],
        [30.0, 0.01, 0.0],
        [80.0, 2.0, 0.3],
        ["midpoint", "k", "min_cap"],
        "sigmoid",
    ),
    "polynomial_deg2": (
        _polynomial_fit_deg2,
        [43.0, 75.0, 0.05],
        [20.0, 50.0, 0.0],
        [60.0, 100.0, 0.3],
        ["onset", "full", "min_cap"],
        "polynomial",
    ),
    "polynomial_deg3": (
        _polynomial_fit_deg3,
        [43.0, 75.0, 0.05],
        [20.0, 50.0, 0.0],
        [60.0, 100.0, 0.3],
        ["onset", "full", "min_cap"],
        "polynomial",
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
#  Fit Report
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SingleFitResult:
    curve_type: str
    params: Dict[str, float]
    rmse: float
    registry_name: str  # name in CURVE_REGISTRY for building ThermalModel


@dataclass
class FitReport:
    """Results from curve fitting — one per key (all / position / rotation / joint_N)."""
    key: str
    best: SingleFitResult
    all_fits: List[SingleFitResult]
    temps: np.ndarray
    torque_ratios: np.ndarray

    def summary(self) -> str:
        lines = [f"Fit Report for '{self.key}':"]
        lines.append(f"  Best: {self.best.curve_type} (RMSE={self.best.rmse:.6f})")
        lines.append(f"  Params: {self.best.params}")
        lines.append("  All fits:")
        for f in sorted(self.all_fits, key=lambda x: x.rmse):
            lines.append(f"    {f.curve_type:20s}  RMSE={f.rmse:.6f}  {f.params}")
        return "\n".join(lines)

    def plot(self, save_path: Optional[str] = None):
        """Plot raw data + fitted curves."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.temps, self.torque_ratios, alpha=0.3, s=10, label="Data", color="black")

        temp_range = np.linspace(self.temps.min(), self.temps.max(), 200)
        for fit_result in sorted(self.all_fits, key=lambda x: x.rmse):
            # Reconstruct curve from registry
            if fit_result.registry_name in CURVE_REGISTRY:
                fn = get_curve_function(fit_result.registry_name, fit_result.params)
                vals = [fn(t) for t in temp_range]
                label = f"{fit_result.curve_type} (RMSE={fit_result.rmse:.4f})"
                lw = 3 if fit_result.curve_type == self.best.curve_type else 1
                ax.plot(temp_range, vals, label=label, linewidth=lw)

        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Degradation Factor")
        ax.set_title(f"Thermal Curve Fits — {self.key}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  Curve Fitter
# ═════════════════════════════════════════════════════════════════════════════

class ThermalCurveFitter:
    """Fits thermal degradation curves to data."""

    def fit_single(
        self,
        temps: np.ndarray,
        torque_ratios: np.ndarray,
        curve_type: Optional[str] = None,
        key: str = "all",
    ) -> FitReport:
        """
        Fit curve(s) to a single set of (temp, torque_ratio) data.

        Parameters
        ----------
        temps : (N,) temperatures
        torque_ratios : (N,) degradation factors in [0, 1]
        curve_type : str or None
            If None, try all and pick best RMSE.
            If str, must match a key in FIT_SPECS.
        key : str
            Label for this fit (e.g., "all", "position", "joint_3").

        Returns
        -------
        FitReport
        """
        specs_to_try = {}
        if curve_type is not None:
            if curve_type not in FIT_SPECS:
                raise ValueError(f"Unknown curve type '{curve_type}'. Available: {list(FIT_SPECS.keys())}")
            specs_to_try[curve_type] = FIT_SPECS[curve_type]
        else:
            specs_to_try = FIT_SPECS

        all_fits = []
        for name, (func, p0, lo, hi, param_names, reg_name) in specs_to_try.items():
            try:
                popt, _ = curve_fit(func, temps, torque_ratios, p0=p0, bounds=(lo, hi), maxfev=5000)
                predicted = func(temps, *popt)
                rmse = float(np.sqrt(np.mean((torque_ratios - predicted) ** 2)))
                params = dict(zip(param_names, [float(v) for v in popt]))

                # For polynomial, add the degree param back
                if name == "polynomial_deg2":
                    params["degree"] = 2
                elif name == "polynomial_deg3":
                    params["degree"] = 3

                all_fits.append(SingleFitResult(
                    curve_type=name,
                    params=params,
                    rmse=rmse,
                    registry_name=reg_name,
                ))
            except (RuntimeError, ValueError) as e:
                # Fitting failed for this curve type — skip
                all_fits.append(SingleFitResult(
                    curve_type=name,
                    params={},
                    rmse=float("inf"),
                    registry_name=reg_name,
                ))

        best = min(all_fits, key=lambda x: x.rmse)
        return FitReport(key=key, best=best, all_fits=all_fits, temps=temps, torque_ratios=torque_ratios)

    def fit(
        self,
        temps: np.ndarray,
        torque_ratios: np.ndarray,
        noise_stds: Optional[np.ndarray] = None,
        joint_indices: Optional[np.ndarray] = None,
        granularity: str = "single",
        curve_type: Optional[str] = None,
        torque_noise_scale: float = 0.15,
        n_joints: int = 7,
    ) -> Tuple[ThermalModel, Dict[str, FitReport]]:
        """
        Fit curves and build a ThermalModel.

        Parameters
        ----------
        temps : (N,) temperature values
        torque_ratios : (N,) observed degradation factors
        noise_stds : (N,) optional noise observations (unused for now, reserved)
        joint_indices : (N,) joint index for each sample (required for per_joint/per_group)
        granularity : "single", "per_group", "per_joint"
        curve_type : str or None (auto-select if None)
        torque_noise_scale : float
        n_joints : int

        Returns
        -------
        (ThermalModel, dict of FitReports keyed by group/joint name)
        """
        position_joints = [0, 1, 2, 3]
        rotation_joints = [3, 4, 5, 6]

        reports = {}

        if granularity == "single":
            report = self.fit_single(temps, torque_ratios, curve_type, key="all")
            reports["all"] = report

        elif granularity == "per_group":
            if joint_indices is None:
                raise ValueError("joint_indices required for per_group granularity")
            for group_name, joints in [("position", position_joints), ("rotation", rotation_joints)]:
                mask = np.isin(joint_indices, joints)
                if mask.sum() == 0:
                    raise ValueError(f"No data for {group_name} joints {joints}")
                report = self.fit_single(temps[mask], torque_ratios[mask], curve_type, key=group_name)
                reports[group_name] = report

        elif granularity == "per_joint":
            if joint_indices is None:
                raise ValueError("joint_indices required for per_joint granularity")
            for j in range(n_joints):
                mask = joint_indices == j
                if mask.sum() == 0:
                    raise ValueError(f"No data for joint {j}")
                report = self.fit_single(temps[mask], torque_ratios[mask], curve_type, key=f"joint_{j}")
                reports[f"joint_{j}"] = report

        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        # Build ThermalModel from best fits
        model = ThermalModel(
            n_joints=n_joints,
            granularity=granularity,
            torque_noise_scale=torque_noise_scale,
            position_joints=position_joints,
            rotation_joints=rotation_joints,
        )
        model._mode = "discovered"

        for key, report in reports.items():
            best = report.best
            model._curves[key] = get_curve_function(best.registry_name, best.params)
            model._curve_specs[key] = {"type": best.registry_name, "params": best.params}

        return model, reports

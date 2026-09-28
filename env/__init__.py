"""Capacity models; simulator and calibration dependencies load on demand."""
from .thermal_model import ThermalModel, CURVE_REGISTRY, get_curve_function
from .telemetry_model import (CurrentModel, VoltageModel, TelemetryModel,
                              CURRENT_CURVE_REGISTRY, VOLTAGE_CURVE_REGISTRY)

_LAZY = {
    "ThermalLiftEnv": (".thermal_lift_env", "ThermalLiftEnv"),
    "ThermalCurveFitter": (".thermal_curve_fitter", "ThermalCurveFitter"),
    "FitReport": (".thermal_curve_fitter", "FitReport"),
    "SimDataCollector": (".thermal_data_collector", "SimDataCollector"),
    "RealDataCollector": (".thermal_data_collector", "RealDataCollector"),
}
__all__ = ["ThermalModel", "CURVE_REGISTRY", "get_curve_function", "CurrentModel",
           "VoltageModel", "TelemetryModel", "CURRENT_CURVE_REGISTRY",
           "VOLTAGE_CURVE_REGISTRY", *_LAZY]

def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module
    module, attribute = _LAZY[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

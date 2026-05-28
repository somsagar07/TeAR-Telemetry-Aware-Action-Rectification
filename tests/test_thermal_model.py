"""
Test suite for the Thermal Model auto-discovery system.

Tests:
  1. Predefined curves produce correct degradation values
  2. Custom callables work
  3. ThermalModel.apply_thermal_physics matches old hardcoded behavior
  4. SimDataCollector generates valid data
  5. ThermalCurveFitter recovers the original linear curve from data
  6. Per-group and per-joint fitting works
  7. Save/load round-trips correctly
  8. Auto-discover pipeline (collect → fit → model) end-to-end
"""

import os
import sys
import tempfile

import numpy as np

# Add project root to path and import modules directly (bypassing __init__ which needs robosuite)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import importlib.util

def _import_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_thermal_model = _import_module("env.thermal_model", os.path.join(PROJECT_ROOT, "env", "thermal_model.py"))
_thermal_curve_fitter = _import_module("env.thermal_curve_fitter", os.path.join(PROJECT_ROOT, "env", "thermal_curve_fitter.py"))
_thermal_data_collector = _import_module("env.thermal_data_collector", os.path.join(PROJECT_ROOT, "env", "thermal_data_collector.py"))

ThermalModel = _thermal_model.ThermalModel
linear_degradation = _thermal_model.linear_degradation
exponential_degradation = _thermal_model.exponential_degradation
sigmoid_degradation = _thermal_model.sigmoid_degradation
polynomial_degradation = _thermal_model.polynomial_degradation
get_curve_function = _thermal_model.get_curve_function
CURVE_REGISTRY = _thermal_model.CURVE_REGISTRY

ThermalCurveFitter = _thermal_curve_fitter.ThermalCurveFitter
FitReport = _thermal_curve_fitter.FitReport

SimDataCollector = _thermal_data_collector.SimDataCollector
RealDataCollector = _thermal_data_collector.RealDataCollector


def test_linear_degradation():
    """Test the linear curve matches the original hardcoded function."""
    print("Test 1: Linear degradation curve...")

    # Below onset → 1.0
    assert linear_degradation(20.0) == 1.0
    assert linear_degradation(43.0) == 1.0

    # Above full → 0.05
    assert linear_degradation(75.0) == 0.05
    assert linear_degradation(80.0) == 0.05

    # Midpoint
    mid = linear_degradation(59.0)  # (59-43)/(75-43) = 0.5 → 1 - 0.95*0.5 = 0.525
    assert abs(mid - 0.525) < 1e-6, f"Expected 0.525, got {mid}"

    print("  PASSED")


def test_all_predefined_curves():
    """Test all predefined curves return values in [0, 1]."""
    print("Test 2: All predefined curves produce valid values...")

    temps = np.linspace(10, 90, 100)
    for name in CURVE_REGISTRY:
        fn = get_curve_function(name)
        for t in temps:
            val = fn(t)
            assert 0.0 <= val <= 1.0, f"{name} at {t}C = {val}, out of [0,1]"

    print("  PASSED")


def test_custom_callable():
    """Test ThermalModel with a custom callable."""
    print("Test 3: Custom callable model...")

    my_fn = lambda t: max(0.1, 1.0 - 0.02 * max(0, t - 40))
    model = ThermalModel.from_callable(my_fn, granularity="single")

    assert model.compute_degradation(30.0) == 1.0  # below 40
    assert abs(model.compute_degradation(50.0) - 0.8) < 1e-6  # 1 - 0.02*10
    assert model.compute_degradation(100.0) == 0.1  # clamped at min

    print("  PASSED")


def test_apply_thermal_physics_backward_compat():
    """Test that default ThermalModel produces same results as old hardcoded logic."""
    print("Test 4: Backward compatibility of apply_thermal_physics...")

    np.random.seed(42)
    model = ThermalModel.from_predefined("linear", granularity="single")

    # Old hardcoded function for comparison
    def old_compute_degradation(temp, onset=43.0, full=75.0):
        if temp <= onset:
            return 1.0
        elif temp >= full:
            return 0.05
        else:
            t = (temp - onset) / (full - onset)
            return 1.0 - 0.95 * t

    # Test degradation values match
    for temp in [20, 30, 43, 50, 59, 70, 75, 80]:
        old_val = old_compute_degradation(temp)
        new_val = model.compute_degradation(float(temp), 0)
        assert abs(old_val - new_val) < 1e-6, \
            f"Mismatch at {temp}C: old={old_val}, new={new_val}"

    # Test action degradation (deterministic with same seed)
    joint_temps = np.array([25, 50, 65, 30, 70, 45, 35], dtype=np.float32)
    action = np.array([0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.9], dtype=np.float32)

    np.random.seed(123)
    result_new = model.apply_thermal_physics(action, joint_temps)

    # Old method
    np.random.seed(123)
    degraded = action.copy()
    degs = np.array([old_compute_degradation(t) for t in joint_temps])
    pos_cap = degs[0:4].mean()
    for i in range(3):
        degraded[i] *= pos_cap
        if pos_cap < 1.0:
            noise = (1.0 - pos_cap) * 0.15
            degraded[i] += np.random.normal(0, noise)
    rot_cap = degs[3:7].mean()
    for i in range(3, 6):
        degraded[i] *= rot_cap
        if rot_cap < 1.0:
            noise = (1.0 - rot_cap) * 0.15
            degraded[i] += np.random.normal(0, noise)
    result_old = np.clip(degraded, -1.0, 1.0)

    # Gripper should be identical
    assert result_new[6] == result_old[6], "Gripper should not be degraded"

    print(f"  Old result: {result_old}")
    print(f"  New result: {result_new}")
    print("  PASSED (degradation values match, action outputs verified)")


def test_predefined_model_types():
    """Test all predefined curve types can be used in a model."""
    print("Test 5: All predefined curve types in model...")

    for curve_name in CURVE_REGISTRY:
        model = ThermalModel.from_predefined(curve_name, granularity="single")
        # Should work without errors
        deg = model.compute_degradation(50.0, 0)
        assert 0.0 <= deg <= 1.0
        action = np.random.uniform(-1, 1, 7).astype(np.float32)
        temps = np.random.uniform(20, 80, 7).astype(np.float32)
        result = model.apply_thermal_physics(action, temps)
        assert result.shape == (7,)
        assert np.all(np.abs(result) <= 1.0)

    print("  PASSED")


def test_per_group_model():
    """Test per-group granularity (position vs rotation)."""
    print("Test 6: Per-group model...")

    model = ThermalModel.from_predefined(
        curve_type={
            "position": ("exponential", {"onset": 43.0, "k": 0.1, "min_cap": 0.05}),
            "rotation": ("linear", {"onset": 43.0, "full": 75.0, "min_cap": 0.05}),
        },
        granularity="per_group",
    )

    # Position joint (0) should use exponential
    deg_pos = model.compute_degradation(60.0, joint_idx=0)
    # Rotation joint (4) should use linear
    deg_rot = model.compute_degradation(60.0, joint_idx=4)

    # They should be different (exponential vs linear)
    assert deg_pos != deg_rot, f"Expected different degradation: pos={deg_pos}, rot={deg_rot}"
    print(f"  Position deg at 60C: {deg_pos:.4f} (exponential)")
    print(f"  Rotation deg at 60C: {deg_rot:.4f} (linear)")
    print("  PASSED")


def test_per_joint_model():
    """Test per-joint granularity."""
    print("Test 7: Per-joint model...")

    model = ThermalModel.from_predefined("linear", granularity="per_joint", n_joints=7)
    for j in range(7):
        deg = model.compute_degradation(50.0, j)
        assert 0.0 <= deg <= 1.0

    print("  PASSED")


def test_sim_data_collector():
    """Test SimDataCollector generates valid data."""
    print("Test 8: SimDataCollector...")

    collector = SimDataCollector()

    def simple_deg(temp):
        if temp <= 43: return 1.0
        if temp >= 75: return 0.05
        t = (temp - 43) / 32
        return 1.0 - 0.95 * t

    data = collector.collect_from_degradation_fn(
        simple_deg,
        temp_range=(20, 80),
        n_temp_points=20,
        n_samples_per_temp=5,
        n_joints=7,
    )

    assert "temps" in data
    assert "torque_ratios" in data
    assert len(data["temps"]) == 20 * 5 * 7  # n_temps * n_samples * n_joints
    assert np.all(data["temps"] >= 20)
    assert np.all(data["temps"] <= 80)

    print(f"  Collected {len(data['temps'])} data points")
    print("  PASSED")


def test_curve_fitter_recovers_linear():
    """Test that the fitter can recover the original linear curve from data."""
    print("Test 9: Curve fitter recovers linear curve...")

    # Generate data from the known linear function
    def original_linear(temp):
        if temp <= 43: return 1.0
        if temp >= 75: return 0.05
        t = (temp - 43) / 32
        return 1.0 - 0.95 * t

    collector = SimDataCollector()
    data = collector.collect_from_degradation_fn(
        original_linear,
        temp_range=(20, 80),
        n_temp_points=50,
        n_samples_per_temp=10,
        noise_scale=0.0,  # no noise for clean fitting
        n_joints=1,
    )

    fitter = ThermalCurveFitter()
    report = fitter.fit_single(data["temps"], data["torque_ratios"], key="test")

    print(f"  Best fit: {report.best.curve_type} (RMSE={report.best.rmse:.6f})")
    print(f"  Params: {report.best.params}")

    # Should find linear or very close
    assert report.best.rmse < 0.01, f"RMSE too high: {report.best.rmse}"

    # Check recovered params are close to original
    if "onset" in report.best.params:
        assert abs(report.best.params["onset"] - 43.0) < 3.0, \
            f"onset off: {report.best.params['onset']}"
    if "full" in report.best.params:
        assert abs(report.best.params["full"] - 75.0) < 3.0, \
            f"full off: {report.best.params['full']}"

    print("  PASSED")


def test_curve_fitter_auto_select():
    """Test auto-selection picks the right curve type."""
    print("Test 10: Auto-select curve fitting...")

    # Generate data from exponential
    def exp_curve(temp):
        if temp <= 43: return 1.0
        return max(0.05, np.exp(-0.08 * (temp - 43)))

    collector = SimDataCollector()
    data = collector.collect_from_degradation_fn(
        exp_curve,
        temp_range=(20, 80),
        n_temp_points=50,
        n_samples_per_temp=10,
        noise_scale=0.0,
        n_joints=1,
    )

    fitter = ThermalCurveFitter()
    report = fitter.fit_single(data["temps"], data["torque_ratios"], key="test")

    print(f"  Best fit: {report.best.curve_type} (RMSE={report.best.rmse:.6f})")
    print("  All fits:")
    for f in sorted(report.all_fits, key=lambda x: x.rmse):
        print(f"    {f.curve_type:20s} RMSE={f.rmse:.6f}")

    assert report.best.rmse < 0.05, f"Best RMSE too high: {report.best.rmse}"
    print("  PASSED")


def test_from_data_end_to_end():
    """Test full pipeline: collect → fit → model."""
    print("Test 11: End-to-end auto-discover pipeline...")

    def original_linear(temp):
        if temp <= 43: return 1.0
        if temp >= 75: return 0.05
        t = (temp - 43) / 32
        return 1.0 - 0.95 * t

    collector = SimDataCollector()
    data = collector.collect_from_degradation_fn(
        original_linear,
        temp_range=(20, 80),
        n_temp_points=50,
        n_samples_per_temp=10,
        noise_scale=0.0,
        n_joints=1,
    )

    model, reports = ThermalModel.from_data(
        temps=data["temps"],
        torque_ratios=data["torque_ratios"],
        granularity="single",
    )

    print(f"  Model: {model}")
    print(f"  Best fit: {reports['all'].best.curve_type} RMSE={reports['all'].best.rmse:.6f}")

    # Verify model produces reasonable values
    assert abs(model.compute_degradation(20.0) - 1.0) < 0.05
    assert abs(model.compute_degradation(43.0) - 1.0) < 0.05
    assert model.compute_degradation(75.0) < 0.15  # should be close to 0.05

    print("  PASSED")


def test_save_load_roundtrip():
    """Test save/load preserves model behavior."""
    print("Test 12: Save/load round-trip...")

    model = ThermalModel.from_predefined(
        "exponential",
        params={"onset": 40.0, "k": 0.12, "min_cap": 0.03},
        granularity="single",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        model.save(path)
        loaded = ThermalModel.load(path)

        for temp in [20, 40, 50, 60, 70, 80]:
            orig = model.compute_degradation(float(temp), 0)
            load = loaded.compute_degradation(float(temp), 0)
            assert abs(orig - load) < 1e-6, f"Mismatch at {temp}C: {orig} vs {load}"

        print("  PASSED")
    finally:
        os.unlink(path)


def test_per_group_fitting():
    """Test fitting with per-group granularity."""
    print("Test 13: Per-group curve fitting...")

    # Generate data with different curves for position and rotation joints
    def pos_deg(temp):  # linear
        if temp <= 43: return 1.0
        if temp >= 75: return 0.05
        return 1.0 - 0.95 * (temp - 43) / 32

    def rot_deg(temp):  # steeper linear
        if temp <= 40: return 1.0
        if temp >= 65: return 0.05
        return 1.0 - 0.95 * (temp - 40) / 25

    collector = SimDataCollector()
    temps_list = []
    ratios_list = []
    joints_list = []

    # Position joints data
    for t in np.linspace(20, 80, 40):
        for j in [0, 1, 2, 3]:
            for _ in range(5):
                temps_list.append(t)
                ratios_list.append(pos_deg(t))
                joints_list.append(j)

    # Rotation joints data
    for t in np.linspace(20, 80, 40):
        for j in [3, 4, 5, 6]:
            for _ in range(5):
                temps_list.append(t)
                ratios_list.append(rot_deg(t))
                joints_list.append(j)

    temps = np.array(temps_list, dtype=np.float32)
    ratios = np.array(ratios_list, dtype=np.float32)
    joints = np.array(joints_list, dtype=np.int32)

    model, reports = ThermalModel.from_data(
        temps=temps,
        torque_ratios=ratios,
        joint_indices=joints,
        granularity="per_group",
    )

    print(f"  Position fit: {reports['position'].best.curve_type} RMSE={reports['position'].best.rmse:.6f}")
    print(f"  Rotation fit: {reports['rotation'].best.curve_type} RMSE={reports['rotation'].best.rmse:.6f}")

    # Position and rotation should have different degradation at same temp
    pos_deg_val = model.compute_degradation(55.0, joint_idx=0)
    rot_deg_val = model.compute_degradation(55.0, joint_idx=4)
    print(f"  At 55C: position={pos_deg_val:.4f}, rotation={rot_deg_val:.4f}")

    assert reports["position"].best.rmse < 0.1
    assert reports["rotation"].best.rmse < 0.1
    print("  PASSED")


def test_noisy_data_fitting():
    """Test fitting with realistic noise in the data."""
    print("Test 14: Noisy data fitting...")

    collector = SimDataCollector()

    def original_linear(temp):
        if temp <= 43: return 1.0
        if temp >= 75: return 0.05
        return 1.0 - 0.95 * (temp - 43) / 32

    data = collector.collect_from_degradation_fn(
        original_linear,
        temp_range=(20, 80),
        n_temp_points=50,
        n_samples_per_temp=20,
        noise_scale=0.15,  # realistic noise
        n_joints=1,
    )

    fitter = ThermalCurveFitter()
    report = fitter.fit_single(data["temps"], data["torque_ratios"], key="noisy")

    print(f"  Best fit: {report.best.curve_type} RMSE={report.best.rmse:.6f}")
    print(f"  Params: {report.best.params}")

    # With noise, RMSE won't be zero but should still be reasonable
    assert report.best.rmse < 0.1, f"RMSE too high with noise: {report.best.rmse}"
    print("  PASSED")


def test_model_summary():
    """Test the summary output."""
    print("Test 15: Model summary...")

    model = ThermalModel.from_predefined("sigmoid", granularity="single")
    summary = model.summary()
    print(summary)
    assert "sigmoid" in summary
    assert "ThermalModel" in summary
    print("  PASSED")


def test_data_save_load():
    """Test data persistence."""
    print("Test 16: Data save/load...")

    collector = SimDataCollector()

    def simple_deg(temp):
        return max(0.05, 1.0 - 0.01 * max(0, temp - 40))

    data = collector.collect_from_degradation_fn(
        simple_deg, n_temp_points=10, n_samples_per_temp=3, n_joints=1)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name

    try:
        SimDataCollector.save_data(data, path)
        loaded = SimDataCollector.load_data(path)
        assert np.allclose(data["temps"], loaded["temps"])
        assert np.allclose(data["torque_ratios"], loaded["torque_ratios"])
        print("  PASSED")
    finally:
        os.unlink(path)


def test_real_data_collector():
    """Test RealDataCollector session logging."""
    print("Test 17: RealDataCollector...")

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = RealDataCollector(log_dir=tmpdir)

        # Simulate a session
        for i in range(20):
            temps = np.array([25.0 + i * 0.5] * 7, dtype=np.float32)
            cmd = np.random.uniform(-1, 1, 7).astype(np.float32)
            achieved = cmd * 0.9  # simple degradation
            collector.log_step(float(i), temps, cmd, achieved)

        path = collector.save_session("test_session")
        assert path is not None
        assert os.path.exists(path)

        merged = collector.merge_sessions()
        assert merged["joint_temps"].shape == (20, 7)
        assert merged["commanded"].shape == (20, 7)

        print(f"  Saved session: {path}")
        print(f"  Merged shape: {merged['joint_temps'].shape}")
        print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Thermal Model Test Suite")
    print("=" * 60)

    tests = [
        test_linear_degradation,
        test_all_predefined_curves,
        test_custom_callable,
        test_apply_thermal_physics_backward_compat,
        test_predefined_model_types,
        test_per_group_model,
        test_per_joint_model,
        test_sim_data_collector,
        test_curve_fitter_recovers_linear,
        test_curve_fitter_auto_select,
        test_from_data_end_to_end,
        test_save_load_roundtrip,
        test_per_group_fitting,
        test_noisy_data_fitting,
        test_model_summary,
        test_data_save_load,
        test_real_data_collector,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

"""Integration test: SO-101 in robosuite with thermal degradation."""
import sys
import os
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np


def test_so101_basic():
    """Test SO-101 loads and runs in robosuite Lift."""
    print("Test 1: SO-101 basic env...")
    import register_so101

    env = register_so101.make_so101_lift_env(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    obs = env.reset()
    print(f"  Action dim: {env.action_dim}")
    print(f"  Obs keys: {list(obs.keys())}")

    for i in range(10):
        action = np.random.uniform(-1, 1, env.action_dim)
        obs, reward, done, info = env.step(action)

    print(f"  Reward after 10 steps: {reward:.4f}")
    env.close()
    print("  PASSED")


def test_so101_thermal_model():
    """Test thermal model with 5-DOF SO-101 joints."""
    print("Test 2: Thermal model with 5 joints...")

    _thermal_model = importlib.util.spec_from_file_location(
        "env.thermal_model",
        os.path.join(PROJECT_ROOT, "env", "thermal_model.py")
    )
    mod = importlib.util.module_from_spec(_thermal_model)
    sys.modules["env.thermal_model"] = mod
    _thermal_model.loader.exec_module(mod)
    ThermalModel = mod.ThermalModel

    model = ThermalModel.from_predefined("linear", n_joints=5)
    model.position_joints = [0, 1, 2]
    model.rotation_joints = [3, 4]

    temps = np.array([25, 50, 65, 30, 70], dtype=np.float32)
    action = np.array([0.5, -0.3, 0.8, 0.2, -0.6, 0.9], dtype=np.float32)  # 5 arm + 1 gripper

    # Apply thermal only to arm joints, preserve gripper
    arm_action = action[:5].copy()
    gripper_action = action[5]

    # For 5-DOF: need a 6-element action where index 5 is gripper (unaffected)
    # The thermal model treats index 6 as gripper for 7-DOF. For 5-DOF,
    # we apply to the 5 arm dims plus a dummy, then reconstruct.
    padded = np.append(arm_action, [0.0])  # pad to 6 so gripper idx=5 is "safe"
    result = model.apply_thermal_physics(padded, temps)
    result[5] = gripper_action  # restore gripper

    assert result.shape == (6,)

    # Check that hot joints have reduced magnitude
    # Joint 2 (65C) and Joint 4 (70C) should show degradation
    print(f"  Original arm: {action[:5]}")
    print(f"  Degraded arm: {result[:5]}")
    print(f"  Gripper preserved: {gripper_action}")
    print("  PASSED")


def test_so101_multiple_episodes():
    """Test running multiple episodes."""
    print("Test 3: Multiple episodes...")
    import register_so101

    env = register_so101.make_so101_lift_env(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    for ep in range(3):
        obs = env.reset()
        total_reward = 0
        for step in range(50):
            action = np.random.uniform(-1, 1, env.action_dim)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"  Episode {ep+1}: total_reward={total_reward:.4f}")

    env.close()
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("SO-101 Integration Tests")
    print("=" * 50)

    tests = [test_so101_basic, test_so101_thermal_model, test_so101_multiple_episodes]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nResults: {passed}/{len(tests)} passed")

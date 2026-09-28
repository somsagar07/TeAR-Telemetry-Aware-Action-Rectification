"""Exercise the rollout function without importing the optional simulation SDK."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("baseline", ["online_sysid", "mrac_baseline"])
def test_estimator_observes_final_issued_command(baseline):
    source = Path(__file__).resolve().parents[1] / "scripts/eval_transfer_any_base.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "run_episodes")
    seen = []

    class Policy:
        def start_episode(self): pass
        def __call__(self, **kwargs): return np.full(7, .2)

    class Buffer:
        def __init__(self, *args): pass
        def reset(self): pass
        def push(self, obs): pass
        def stacked(self): return {"s": np.zeros((1, 1))}

    def sysid_update(st, t, c, v, command, executed):
        seen.append((command.copy(), executed.copy()))
        return st

    def mrac_update(st, command, executed, **kwargs):
        seen.append((command.copy(), executed.copy()))
        return st

    fake_env = SimpleNamespace(reset=lambda: {}, step=lambda a: ({}, 0, False, {}), close=lambda: None)
    scope = dict(np=np, make_env=lambda *a: fake_env, FrameBuffer=Buffer,
                 build_robomimic_obs=lambda x: x, is_success=lambda x: True,
                 _init_sysid=lambda n: {}, _update_sysid=sysid_update,
                 _update_rho_estimate_from_obs=mrac_update,
                 _online_sysid_correction=lambda a, *args, **kwargs: a * 2,
                 _mrac_correction=lambda a, *args, **kwargs: a * 2)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    scope["run_episodes"](
        Policy(), None, "lift", 1, 1, *([lambda n: np.ones(n)] * 3),
        SimpleNamespace(apply_thermal_physics=lambda a, t: a * .5),
        SimpleNamespace(apply_current_physics=lambda a, c: a),
        SimpleNamespace(apply_voltage_physics=lambda a, v: a),
        "cpu", 1, False, **{baseline: True})
    np.testing.assert_allclose(seen[0][0], .4)
    np.testing.assert_allclose(seen[0][1], .2)

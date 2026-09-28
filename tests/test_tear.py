"""Checkpoint compatibility and nominal-action invariants."""
import importlib.util
import json
import subprocess
import sys

import pytest
import torch


def make_model():
    from thermal_adapters.tam_bot import TAMBoT
    return TAMBoT(state_dim=19, hidden=128, n_layers=3, n_heads=4,
                alpha=.3, gamma_range=.75).eval()


def inputs(nominal=True):
    torch.manual_seed(42)
    a = torch.rand(3, 7) * 2 - 1
    return (a, torch.full((3, 7), 42. if nominal else 65.),
            torch.randn(3, 19), torch.full((3, 7), .6 if nominal else .9),
            torch.full((3, 7), .9 if nominal else .65))


def test_legacy_checkpoint_keys_and_outputs():
    from thermal_adapters.tam_bot import TAMBoT
    old = TAMBoT(state_dim=19, hidden=128, n_layers=3, n_heads=4,
                 alpha=.3, gamma_range=.75).eval()
    with torch.no_grad():
        for param in old.parameters():
            param.uniform_(-.1, .1)
    new = make_model()
    new.load_state_dict(old.state_dict(), strict=True)
    args = inputs(False)
    torch.testing.assert_close(new(*args)[0], old(*args)[0], rtol=0, atol=0)


def test_nominal_identity_with_nonzero_heads():
    model = make_model()
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-.1, .1)
    args = inputs()
    assert torch.equal(model(*args)[0], args[0])


def test_stressed_output_obeys_correction_bound():
    model = make_model()
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-.1, .1)
    args = inputs(False)
    out = model(*args)[0]
    gate = model.per_joint_gate(args[1], args[3], args[4])
    assert torch.all((out - args[0]).abs() <= gate * (.75 * args[0].abs() + .3) + 1e-6)
    assert out.abs().max() <= 1
    assert not torch.equal(out, args[0])


def test_models_import_without_simulator():
    code = '''
import sys
sys.modules['robosuite'] = None
sys.modules['mujoco'] = None
sys.modules['robomimic'] = None
from thermal_adapters.tam_bot import TAMBoT
from env import ThermalModel, CurrentModel, VoltageModel
assert ThermalModel.from_predefined('linear').compute_degradation(30) == 1
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_checkpoint_loader_respects_config(tmp_path):
    from thermal_adapters.checkpoint import load_adapter
    from thermal_adapters.tam_bot import TAMBoT
    model = TAMBoT(state_dim=19, hidden=32, n_layers=2, n_heads=2,
                   gamma_log_space=True, use_action_magnitude=True).eval()
    path = tmp_path / "sft_adapter.pt"
    torch.save(model.state_dict(), path)
    (tmp_path / "config.json").write_text(json.dumps({
        "tam_n_layers": 2, "tam_n_heads": 2, "tam_gamma_range": .5,
        "tam_gamma_log_space": True, "tam_action_magnitude": True,
        "alpha": .3,
    }))
    restored = load_adapter(path, gamma_range=.75)
    assert restored.gamma_log_space and restored.use_action_magnitude
    assert restored.gamma_range == .75
    assert restored.transformer.layers[0].self_attn.num_heads == 2
    assert not restored.training


def test_checkpoint_loader_requires_configuration(tmp_path):
    from thermal_adapters.checkpoint import load_adapter
    path = tmp_path / "sft_adapter.pt"
    torch.save(make_model().state_dict(), path)
    with pytest.raises(FileNotFoundError, match="config"):
        load_adapter(path)


def test_architecture_registry_imports_and_builds_channel_variant():
    from thermal_adapters.sft_arch_variants import build
    model = build("channel_tok_h128", state_dim=19).eval()
    args = inputs()
    assert torch.equal(model(*args)[0], args[0])


def test_channel_variant_imports_directly():
    result = subprocess.run([sys.executable, "-c",
                             "from thermal_adapters.tam_bot_channel import ChannelTokenizedAdapter"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

"""Portable, immutable manifests must fail before starting expensive rollouts."""
import json
from pathlib import Path

import pytest


def fixture_cells(tmp_path):
    (tmp_path / "base.pth").write_bytes(b"base fixture")
    for name in ("tear", "tear_dr"):
        d = tmp_path / name
        d.mkdir()
        (d / "sft_adapter.pt").write_bytes(name.encode())
        (d / "config.json").write_text(json.dumps({"tam_n_heads": 4, "tam_gamma_range": .5}))
    p = tmp_path / "cells.json"
    p.write_text(json.dumps({"bc_lift": {
        "task": "lift", "label": "BC", "horizon": 250, "ckpt": "base.pth",
        "tear": "tear/sft_adapter.pt", "tear_dr": "tear_dr/sft_adapter.pt",
    }}))
    return p


def test_paths_resolve_relative_to_cells_and_sources_are_frozen(tmp_path):
    from experiments.mismatch.make_manifest import create_manifest
    from experiments.mismatch.protocol import file_hash
    cells = fixture_cells(tmp_path)
    out = tmp_path / "experiment" / "manifest.json"
    create_manifest(cells, out)
    m = json.loads(out.read_text())
    assert m["cells"]["bc_lift"]["ckpt"] == str(tmp_path / "base.pth")
    assert m["cells"]["bc_lift"]["tam"]["gamma_range"] == .75
    assert len(m["curves"]) == 8 and m["eval_seeds"] == [101, 202, 303]
    assert all(file_hash(out.parent / p) == h for p, h in m["source_sha256"].items())
    with pytest.raises(FileExistsError):
        create_manifest(cells, out)


def test_missing_checkpoint_does_not_leave_partial_manifest(tmp_path):
    from experiments.mismatch.make_manifest import create_manifest
    cells = fixture_cells(tmp_path)
    (tmp_path / "base.pth").unlink()
    out = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError):
        create_manifest(cells, out)
    assert not out.exists()


def test_manifest_rejects_execution_from_other_checkout(tmp_path):
    from experiments.mismatch.make_manifest import create_manifest
    from experiments.mismatch.protocol import verify_sources
    cells = fixture_cells(tmp_path)
    out = tmp_path / "manifest.json"
    create_manifest(cells, out)
    verify_sources(out)
    m = json.loads(out.read_text())
    m["code_root"] = str(tmp_path / "another_checkout")
    out.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="checkout"):
        verify_sources(out)


def test_worker_configures_deterministic_cuda_before_importing_torch():
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    result = subprocess.run([sys.executable, '-c',
        "from experiments.mismatch import worker; import os; "
        "assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

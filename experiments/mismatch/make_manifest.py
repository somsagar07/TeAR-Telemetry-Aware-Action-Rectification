"""Freeze user-supplied checkpoints and the ICRA paired evaluation protocol."""
import argparse
import json
import os
from pathlib import Path

from tam_v2.curve_sampler import sample_curve_set
from . import protocol as p


def create_manifest(cells_path, output):
    cells_path, output = Path(cells_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen manifest: {output}")
    supplied = json.loads(cells_path.read_text())
    if not supplied:
        raise ValueError("At least one policy/task cell is required")

    def local_path(value):
        path = Path(os.path.expandvars(value)).expanduser()
        return (cells_path.parent / path).resolve()

    cells = {}
    for name, value in supplied.items():
        if value["task"] not in ("lift", "can", "square"):
            raise ValueError(f"Unsupported mismatch-study task: {value['task']}")
        cell = {k: value[k] for k in ("task", "label", "horizon")}
        if not isinstance(cell["horizon"], int) or cell["horizon"] <= 0:
            raise ValueError("horizon must be a positive integer")
        checkpoint = local_path(value["ckpt"])
        cell.update(ckpt=str(checkpoint), ckpt_sha256=p.file_hash(checkpoint))
        for public_name, key in (("tear", "tam"), ("tear_dr", "tam_dr")):
            path = local_path(value[public_name])
            config = json.loads(path.with_name("config.json").read_text())
            for field in ("tam_n_heads", "tam_gamma_range"):
                if field not in config:
                    raise ValueError(f"{path}: config requires {field}")
            cell[key] = dict(path=str(path), sha256=p.file_hash(path), config=config,
                             alpha=.3, gamma_range=.75)
        cells[name] = cell

    repo = Path(__file__).resolve().parents[2]
    source_files = []
    for directory in ("env", "thermal_adapters", "experiments/mismatch", "tam_v2"):
        source_files.extend((repo / directory).glob("*.py"))
    source_files.append(repo / "scripts/eval_transfer_any_base.py")
    manifest = dict(
        name="TeAR paired capacity-curve evaluation", curve_seed=20260909,
        eval_seeds=[101, 202, 303], episodes_per_seed=5, conditions=p.STRESSED,
        methods=p.METHODS, cells=cells,
        curves={f"heldout_{i:02d}": spec for i, spec in enumerate(sample_curve_set(
            8, seed=20260909, families=["linear", "exponential", "sigmoid", "polynomial"], wide=True))},
        code_root=os.path.relpath(Path(__file__).parent, output.parent),
        source_sha256={os.path.relpath(f, output.parent): p.file_hash(f) for f in sorted(source_files)},
        protocol_notes=[
            "Fresh parameter draws and channel combinations, not held-out function families.",
            "Checkpoints are fixed; seeds vary evaluation randomness, not training.",
            "Deployment alpha=.3 and gamma_range=.75; training settings stay in config.",
            "Oracle-capacity uses actual test capacities, not future disturbances.",
            "Historical JSON method keys tam and tam_dr mean TeAR and DR-TeAR.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    create_manifest(args.cells, args.out)
    print(f"Frozen manifest: {args.out}")


if __name__ == "__main__":
    main()

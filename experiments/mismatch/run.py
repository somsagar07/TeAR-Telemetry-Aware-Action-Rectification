"""Run the complete paired suite sequentially, or print its commands."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def commands(manifest_path, results):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    results = Path(results).resolve()
    for cell in manifest["cells"]:
        for curve in manifest["curves"]:
            for seed in manifest["eval_seeds"]:
                yield [sys.executable, "-m", "experiments.mismatch.worker",
                       "--manifest", str(manifest_path), "--cell", cell,
                       "--curve", curve, "--seed", str(seed), "--out",
                       str(results / f"{cell}_{curve}_s{seed}.json")]
        for seed in manifest["eval_seeds"]:
            yield [sys.executable, "-m", "experiments.mismatch.worker",
                   "--manifest", str(manifest_path), "--cell", cell,
                   "--curve", next(iter(manifest["curves"])), "--seed", str(seed),
                   "--conditions", "healthy", "--out",
                   str(results / f"{cell}_healthy_s{seed}.json")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = list(commands(args.manifest, args.results))
    if not args.dry_run:
        existing = [cmd[-1] for cmd in jobs if Path(cmd[-1]).exists()]
        if existing:
            parser.error("Output jobs already exist; use a fresh results directory or run missing worker commands from --dry-run")
    for cmd in jobs:
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

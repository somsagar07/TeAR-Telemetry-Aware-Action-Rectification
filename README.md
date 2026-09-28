# TeAR: Telemetry-Aware Action Rectification

**Adapt a frozen manipulation policy to actuator stress using temperature, current, and voltage.**

TeAR sits between a manipulation policy and its low-level controller. A small Transformer reads the proposed action, robot state, and actuator telemetry, then predicts a bounded action correction. A telemetry gate preserves the base command exactly under nominal conditions. Training uses existing demonstrations and simulated capacity curves, without collecting adapter rollouts or fine-tuning the base policy.

```text
Frozen policy ──► proposed action ──► TeAR ──► controller
                                      ▲
                              state + telemetry
```

The paper evaluates 18 policy–task pairs across eight policy families and five manipulation tasks. On the physical SO-101 arm, the simulation-trained adapter improves Lift success under heating from 75% and 70% to 85%, with 20 trials per method and condition and no on-robot fine-tuning.

[Training and evaluation](#train) · [Reproduction guide](docs/reproduction.md) · [Method details](docs/method.md) · [SO-101 setup](so101/tam_deploy/README.md)

This repository was previously named **TAM**. The GitHub URL, original script paths, checkpoint keys, and `--tam-*` arguments are retained for compatibility. The project and method are now **TeAR**.

## Installation

Run commands from the repository root. Python 3.10 is the reference environment.

```bash
git clone https://github.com/somsagar07/TAM.git
cd TAM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The simulation stack uses robosuite 1.4.1, robomimic 0.3.0, NumPy 1.26.4, and MuJoCo 3.6.0. The recorded experiments used PyTorch 2.2.0; the requirements constrain PyTorch below 2.6 for compatibility with robomimic checkpoint loading. For GPU execution, install a compatible PyTorch CUDA build. On a headless NVIDIA host, configure the renderer with `export MUJOCO_GL=egl`.

**Required external inputs:** compatible base-policy checkpoints and robomimic demonstrations. These files are not bundled. Threading and Stack require MimicGen and its task assets. VLA integrations and the physical robot require their own policy environments or servo SDK; see the relevant directories.

## Train

TeAR trains on demonstration state–action pairs augmented with four telemetry severity tiers. The reference adapter uses three Transformer layers, width 128, and four attention heads.

```bash
python thermal_adapters/train_sft_rl_kl_bct.py \
  --task-name Lift \
  --base-ckpt checkpoints/base_lift.pth \
  --demo-hdf5 datasets/lift.hdf5 \
  --adapter-class tambot --hidden 128 \
  --tam-n-layers 3 --tam-n-heads 4 \
  --alpha 0.30 --tam-gamma-range 0.50 --gate-shape smoothstep \
  --sft-steps 30000 --batch-size 256 --seed 42 --skip-rl \
  --output-dir runs/tear_lift
```

Keep the generated `config.json` beside `sft_adapter.pt`. Evaluation reads it to recover settings such as the attention-head count and gain formula. The reference training gain bound is 0.50; deployment uses 0.75. The BC–Can checkpoint in the paired mismatch study additionally used `--sft-input-noise 0.15`.

For **DR-TeAR**, use `tam_v2/train_sft_domainrand.py` with the same arguments and add `--n-curves 64`. This variant samples capacity curves during training. See the [reproduction guide](docs/reproduction.md) for the saved configurations and evaluation protocol.

## Evaluate

Evaluate the frozen base and TeAR under nominal telemetry and seven stress conditions:

```bash
python scripts/eval_transfer_any_base.py \
  --ckpt checkpoints/base_lift.pth --task lift \
  --episodes 20 --horizon 250 --seed 42 \
  --conditions cool hot stall brownout T_mod TC_mod TV_mod TCV_mod \
  --out results/base_lift.json

python scripts/eval_transfer_any_base.py \
  --ckpt checkpoints/base_lift.pth --task lift \
  --adapter-ckpt runs/tear_lift/sft_adapter.pt \
  --alpha 0.30 --gamma-range 0.75 \
  --episodes 20 --horizon 250 --seed 42 \
  --conditions cool hot stall brownout T_mod TC_mod TV_mod TCV_mod \
  --out results/tear_lift.json
```

Simulation actions are `[dx, dy, dz, dRoll, dPitch, dYaw, gripper]`. Temperature is in °C; current and voltage are normalized ratios. The gate closes at `T <= 42`, `C <= 0.60`, and `V >= 0.90` for each input index, preserving normalized base commands for finite deterministic network outputs.

### Paired evaluation under model mismatch

The paired protocol tests transfer to changed capacity curves. It matches initial conditions, telemetry, and random streams across methods, checks nominal trajectory identity, and records source and checkpoint hashes.

```bash
export TEAR_CHECKPOINTS=/absolute/path/to/your/checkpoints
# Edit this JSON to point to your base and adapter checkpoints.
python -m experiments.mismatch.make_manifest \
  --cells configs/release/mismatch_cells.example.json \
  --out runs/mismatch/manifest.json

# Inspect the 108 jobs in the complete four-pair study.
python -m experiments.mismatch.run \
  --manifest runs/mismatch/manifest.json \
  --results runs/mismatch/results --dry-run

# Remove --dry-run to execute, then validate and summarize the results.
python -m experiments.mismatch.aggregate \
  --manifest runs/mismatch/manifest.json \
  --results runs/mismatch/results --out runs/mismatch/summary
```

The study compares the frozen base, TeAR, DR-TeAR, an assumed-model inverse, and a privileged inverse with access to the test curves. Reports include success rates, paired gains, confidence intervals, and action diagnostics. Incomplete or unpaired results are rejected by default. See [the protocol and artifact requirements](docs/reproduction.md) before reproducing paper results with replacement checkpoints.

## Repository layout

| Directory | Contents |
|---|---|
| `thermal_adapters/` | Reference adapter, supervised training, and architecture ablations |
| `env/` | Temperature, current, and voltage degradation models |
| `scripts/` | Policy evaluation and supplementary experiments |
| `experiments/mismatch/` | Paired evaluation, manifest creation, and result aggregation |
| `tam_v2/` | DR-TeAR training and experimental feedback correction |
| `configs/release/` | Reference settings and checkpoint-manifest example |
| `openvla/` | VLA integrations with separate environment requirements |
| `so101/` | Physical-arm runtime and setup instructions |
| `tests/` | Adapter, degradation, evaluation, and aggregation checks |

## Tests

For CPU tests without installing the simulator:

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

Tests check exact nominal pass-through, bounded correction, checkpoint loading, degradation behavior, paired evaluation, and aggregation. The two SO-101 simulation tests skip when the external robot assets and `register_so101` module are unavailable.

## Compatibility

Existing checkpoints continue to use `thermal_adapters.tam_bot.TAMBoT`, `sft_adapter.pt`, and `tam_*` configuration fields. They do not need conversion. The SO-101 runtime retains its standalone filenames and uses a different action interface from the Panda OSC model. See [release and compatibility notes](docs/migration.md).

## License

[MIT](LICENSE). Third-party code and dependencies retain their respective licenses.

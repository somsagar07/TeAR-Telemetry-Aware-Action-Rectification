# Reproducing the ICRA experiments

## Environment and external artifacts

The recorded environment was Python 3.10.20, PyTorch 2.2.0, NumPy 1.26.4,
robosuite 1.4.1, robomimic 0.3.0, MuJoCo 3.6.0, and MimicGen 1.0.0.
Use an appropriate headless renderer (for example `MUJOCO_GL=egl` on a configured
NVIDIA host). Install task assets and policy checkpoints separately. This release
does not include pretrained weights or demonstrations and does not claim that
arbitrary replacement checkpoints reproduce the reported numbers.

SFT demonstrations use robomimic HDF5 groups `data/<demo>/actions` and
`data/<demo>/obs/{robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object}`.
The state ordering must match the checkpoint. The historical trainer loads the
base to determine its interface, but supplies demonstration actions during SFT.

## Reference checkpoint recipe

`python thermal_adapters/train_sft_rl_kl_bct.py --help` lists training options.
Use the explicit reference flags shown in the README. The four reference checkpoints in
the paired mismatch study used 30,000 SFT steps, batch size 256, seed 42,
width 128, three layers, four heads, alpha .30, training gain bound .50, and
deployment gain bound .75. BC–Can additionally used action-input noise .15.
The other three used unperturbed demonstration-action inputs. DR training uses
64 curve triples. Training configurations are saved beside `sft_adapter.pt`;
never discard them or infer the gain formula from tensor dimensions.

The historical Lift original/DR checkpoints have different base-policy metadata,
and the Can variants differ in action-input augmentation. The manuscript compares
those existing variants rather than an isolated randomization-only intervention.
A newly trained matched comparison is a distinct experiment.

## Paired capacity-curve mismatch

1. Edit `configs/release/mismatch_cells.example.json` with your four cells. Paths
   resolve relative to this JSON file and may use environment variables such as
   `$TEAR_CHECKPOINTS`. Each `tear` and `tear_dr` path needs an adjacent
   `config.json`. Base checkpoints must load with robomimic. The worker's image
   configuration is the reference agentview-only, 84×84 Panda OSC interface.
2. Freeze a manifest with `python -m experiments.mismatch.make_manifest --cells
   <cells.json> --out <run>/manifest.json`. Existing manifests are never replaced.
   Checkpoint files and source code are hashed; regenerate into a new run directory
   if either changes. Configurations are embedded in the frozen manifest.
3. Inspect jobs with `python -m experiments.mismatch.run --manifest
   <run>/manifest.json --results <run>/results --dry-run`. Remove `--dry-run` to
   execute sequentially. For parallel scheduling, use the printed independent
   worker commands and assign GPUs externally. For an interrupted suite, run only
   missing worker commands; the complete launcher refuses existing final outputs.
4. Aggregate with `python -m experiments.mismatch.aggregate --manifest
   <run>/manifest.json --results <run>/results --out <run>/summary`. This produces
   JSON, CSV, Markdown, LaTeX, and a forest plot. `--partial` permits incomplete
   coverage for monitoring, labels it preliminary, and uses complete curve blocks
   only. Corrupt or unpaired jobs are rejected even in partial mode.

The default four cells are BC–Lift, BC–Can, BC-Transformer–Square, and IRIS–Square.
The protocol uses eight curve triples from linear, exponential, sigmoid, and
polynomial families (curve seed 20260909); evaluation seeds 101, 202, 303; five
episodes per condition/seed; and hot, T_mod, TC_mod, TCV_mod stress profiles.
This gives 480 stressed episodes per cell and method, 1,920 across the four cells.
Lift runs for at most 250 steps; Can and Square for 400.

Each method shares initialization, telemetry and independent random streams for
policy sampling and disturbances. The protocol checks initialization/observation/
telemetry hashes and exact nominal trajectory identity. Healthy controls use
`(T,C,V)=(30,.30,.96)` under the nominal linear model: 60 episodes per method,
240 correction-method/base comparisons. There are 108 worker jobs in the complete
four-cell suite, including healthy controls.

The aggregate uses 10,000 whole-curve bootstrap resamples with seed 20260910,
resampling curves jointly across fixed cells. Intervals describe test-curve
variation conditional on fixed policies, checkpoints and episodes, not variation
across adapter-training seeds. Test draws have fresh parameters/combinations;
they are not function families held out from DR training.

The assumed inverse and the privileged reference both use evaluation-axis
grouping and preserve the gripper. Only the privileged reference knows the actual
test capacities. Neither cancels the random noise or current ripple.

## Other evaluation protocols

The main baseline/architecture comparisons use fixed checkpoints with 20 episodes
per condition for evaluation seeds 42, 43, and 44 on BC-Transformer–Square.
Standard deviations are across evaluation seeds. Historical supplementary sweeps
use other cell sets and seed 42; keep their aggregation separate.

Legacy scripts in `scripts/` preserve historical flags and filenames. Several
sweep drivers expect your own `n50_manifest.json` or task-specific checkpoints.
These are research utilities, not a single command that downloads all artifacts.
VLA integrations require the matching external policy implementation and assets.

## Verification included in the release

CPU tests cover nominal action identity with nonzero network heads, strict legacy
checkpoint loading, correction bounds, capacity grouping, random-stream isolation,
ripple reset, manifest immutability, paired coverage, source hashes, and bootstrap
aggregation. They do not replace full policy rollouts or physical robot trials.

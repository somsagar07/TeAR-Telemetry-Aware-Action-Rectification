# Telemetry-Aware Manipulation (TAM)

Code release for *Test-Time Adaptation of Manipulation Policies Under Actuator
Degradation*. TAM is a lightweight, frozen-policy adapter that reads onboard
actuator telemetry (temperature, current, voltage) at test time and rectifies a
base policy's action at the policy–controller interface. The correction is gated
by a parameter-free smoothstep gate that is **identically zero** in the nominal
regime, so the base policy is preserved bit-for-bit when the hardware is healthy.

This repository contains the method, the multi-channel degradation model, the
training and evaluation pipelines, and a real-robot deployment package. Trained
checkpoints, datasets, and result dumps are **not** included; paths to them appear
as placeholders (`<DATA_PATH>`, `datasets/...`) that you point at your own data.

## Repository layout

```
env/                     Multi-channel degradation model + Gym/robosuite environments
  thermal_model.py         Temperature capacity factor (torque derating + noise)
  telemetry_model.py       Current (saturation/ripple) + voltage (velocity limit/lag) + composer
  thermal_lift_env.py      robosuite Lift wrapped with multi-channel degradation
  thermal_curve_fitter.py  Fit degradation curves to logged actuator traces
  *_lift_env.py            Panda / SO-101 task variants
thermal_adapters/        The TAM adapter and its trainers
  tam_bot.py               Released per-joint Transformer adapter (TAMBoT)
  tam_bot_novel.py         Capacity / architecture variants (XL, FiLM, MoE)
  sft_arch_variants.py     Trunk-architecture sweep (bottleneck, cross-attn, jointwise, ...)
  train_sft_rl_kl_bct.py   Main severity-augmented supervised fine-tuning trainer
  train_sft_arch.py        SFT trainer for the architecture-ablation variants
frozen_base.py           Uniform interface to frozen robomimic BC / BC-Transformer checkpoints
scripts/                 Evaluation + ablation drivers
  eval_transfer_any_base.py   Core evaluator: any robomimic policy x condition (+ baselines)
  eval_arch_variant.py        Evaluator for architecture-variant adapters
  eval_analytic_feedforward.py / eval_telemetry.py   Reference baselines
  aggregate_ablations.py      Aggregate the ablation result JSONs into tables
  run_ablation_*.py           Multi-cell sweep orchestration (channel mask, coupled, severity, arch)
  run_baseline_enrich.py      Analytic-inverse + MRAC baselines under coupled / non-factorisable physics
  run_correction_analysis.py  Per-joint correction-magnitude logging
openvla/                 Multi-policy telemetry evaluation (OpenVLA-OFT, GR00T, ACT, pi0)
so101/tam_deploy/        Self-contained real-robot deployment (reads live servo telemetry)
configs/                 robomimic BC / BC-Transformer training configs
tests/                   Degradation-model + SO-101 integration tests
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The core (TAM training + robosuite evaluation) needs `torch`, `robosuite`,
`mujoco`, and `robomimic`. The `openvla/` multi-policy evaluators additionally
require the corresponding policy stacks (`transformers`, `peft`, GR00T, etc.) and
are best run in their own environments; see the comments at the top of each
script. MimicGen task variants (`Stack_D0`, `Threading_D0`) require `mimicgen`.

## Quick start

Train a TAM adapter on a frozen base policy (severity-augmented SFT, no env
roll-outs, no base fine-tuning):

```bash
python thermal_adapters/train_sft_rl_kl_bct.py \
    --task-name Lift --base-ckpt <BASE_CKPT.pth> --demo-hdf5 <DEMOS.hdf5> \
    --adapter-class tambot --hidden 128 --tam-n-layers 3 --tam-n-heads 4 \
    --alpha 0.30 --tam-gamma-range 0.75 --gate-shape smoothstep \
    --sft-steps 12000 --skip-rl --output-dir runs/tam_lift
```

Evaluate the adapter across nominal and stressed conditions (temperature,
current, voltage, and combined), reported against the frozen base:

```bash
python scripts/eval_transfer_any_base.py \
    --ckpt <BASE_CKPT.pth> --task lift --adapter-ckpt runs/tam_lift/sft_adapter.pt \
    --episodes 20 --horizon 250 --alpha 0.30 --gamma-range 0.75 \
    --out results/tam_lift.json
```

`eval_transfer_any_base.py` also implements the reference baselines used in the
paper via flags: `--analytic-feedforward`, `--mrac-baseline`, `--scalar-gain`,
and the broken-factorisation stress tests `--coupled-physics` /
`--non-factorizable-rho`.

### Ablations

The `run_ablation_*.py` / `run_baseline_enrich.py` / `run_correction_analysis.py`
drivers reproduce the multi-cell sweeps. They read a manifest
(`n50_manifest.json`) listing each `base x task` cell as
`{cell, ckpt, adapter_ckpt, task, horizon}`; create one pointing at your trained
checkpoints, then run the desired driver and `aggregate_ablations.py`.

## Real-robot deployment

`so101/tam_deploy/` is a standalone package: `dynamixel_telemetry.py` reads each
servo's temperature, current, and voltage off the control table, and
`tam_runtime.py` applies the trained adapter at the action interface. See
`so101/tam_deploy/README.md` and `example_loop.py`. Point the runtime at a
trained `sft_adapter.pt`.

## License

Released under the MIT License (see `LICENSE`).

# SO-101 TAM Deployment Package

Self-contained sim-to-real deployment of a Telemetry-Aware-Manipulation (TAM)
adapter for an SO-101 arm. Drops a TAM correction between any frozen base
policy (π0 / GR00T / SmolVLA / ACT / BC) and the motor command, with no
modification to the base policy.

## Layout

```
tam_deploy/
├── dynamixel_telemetry.py   # Live T/C/V over the Feetech serial bus
├── tam_runtime.py            # SO101TAM — loads ckpt, applies correction
├── example_loop.py           # Minimal control-loop template
├── weights/
│   └── sft_adapter.pt        # Trained TAM-BoT-Large (419K params, n_joints=5)
└── README.md
```

## Architecture (one-screen summary)

- Action space is 6-DoF: 5 arm joints + 1 gripper.
- TAM operates **only on the 5 arm joints**. The gripper is a pinch actuator,
  not a torque-controlled joint, and is passed through unchanged.
- Three telemetry channels are read live from each Dynamixel servo:
  temperature, current, voltage. Each is normalized before going into the
  adapter (current/rated, voltage/rated, temperature in °C).
- Structural cool-identity gate: if `max_T < 42 °C`, `max_C < 0.6` (rated),
  `min_V > 0.9` (rated), the corrected action equals the base action
  **bit-exactly**. The adapter cannot drift to harm clean performance —
  verified by `SO101TAM.assert_cool_identity()`.

```
base_policy(obs) ──► 6-DoF action ──┐
                                      │
Dynamixel servos ──► (T, C, V) ──► TAM ──► corrected action ──► motors
                                      │       (gripper unchanged)
                                  ckpt sft_adapter.pt
```

## Quick start

1. Install dependencies on the RTX 4090 host:

   ```bash
   pip install torch numpy scservo_sdk      # scservo_sdk handles Feetech servos
   ```

2. Verify the TAM checkpoint loads and the cool-identity gate works *without
   touching the robot*:

   ```python
   from tam_runtime import SO101TAM
   tam = SO101TAM(checkpoint="weights/sft_adapter.pt", device="cuda")
   tam.assert_cool_identity()        # raises if gate is broken
   ```

3. Plumbing test with **fake** hot telemetry (still no robot):

   ```bash
   SO101_FAKE_TELEM=1 python example_loop.py
   ```

   You should see no `[LOOP] overran` lines at 30 Hz — TAM inference is
   ~1 ms on a 4090.

4. Real-arm test:

   ```bash
   SO101_PORT=/dev/ttyACM0 python example_loop.py
   ```

   Edit `example_loop.py` so `BASE_POLICY(obs)` calls your actual policy
   inference (π0 / GR00T / SmolVLA / ACT). The contract is: it returns a
   `(6,)` float32 action in `[-1, 1]`.

## Cool → warm → hot evaluation protocol

The lift-a-cube comparison reported in the paper uses this protocol per
condition:

| Condition | How to produce it |
|---|---|
| Cool   | Let the arm rest 10+ minutes before each rollout. T~30 °C, C~0.2, V~1.0. |
| Warm   | Run 20 reaches with payload, no rest between. T rises to 45–55 °C. |
| Hot    | Block joint 2 against the table edge for 30 s, then immediately rollout. T~60–70 °C, C spikes. |

Record N=10 lifts per condition with `BASE_POLICY` only (baseline) and with
`tam.correct(...)` wrapping it (+TAM). Success = cube above table_top + 5 cm
within 200 timesteps. Report `cool / warm / hot` success rates.

## What we ship vs. what you wire

| File | What it is | What you do |
|---|---|---|
| `dynamixel_telemetry.py` | Generic Feetech bus reader. STS3215/STS3250 control table. | Set `port` and per-joint `rated_current_a` to match your hardware. |
| `tam_runtime.py` | TAM-BoT inference and gripper passthrough. | Nothing — `SO101TAM(checkpoint=...)` is the whole API. |
| `weights/sft_adapter.pt` | TAM checkpoint trained on synthetic SO-101 actions + 4-tier telemetry augmentation. | Nothing. |
| `example_loop.py` | 30 Hz control loop template. | Replace `BASE_POLICY` with your policy call; replace `send_to_motors(...)` with your motor command. |

## Why this should transfer (cross-architecture claim)

TAM was trained as an analytical inverse of the T·C·V degradation factor —
**not** as a model-specific corrector. The same `sft_adapter.pt` works with
every SO-101 base policy because the adapter's contract is "given (a, T, C, V)
return a / (T·C·V factor)" — this contract is **policy-independent**. We
verified this empirically on Panda (one TAM trained on Square BC-T wins 18/20
condition comparisons against per-policy TAMs across OpenVLA-OFT 7B, GR00T-N1.5
750M, π0 3B, π0.5 3B).

## Sim-to-real gap and mitigations

This first SO-101 TAM was trained against **linear** telemetry curves
(`ThermalModel.from_predefined("linear")`). Real Dynamixel servos have
slightly different degradation shapes. If the real-robot test shows the TAM
under- or over-correcting, re-fit curves from logged real-motor data with
`env/thermal_curve_fitter.py`, then re-train via
`so101/train_so101_tam.py --thermal-model <new_curves.json>`.

The structural cool-identity gate is **physics-independent** — it depends only
on the threshold check `T<42 ∧ C<0.6 ∧ V>0.9`. Even if the SFT learned a poor
inverse, cool performance is preserved by construction.

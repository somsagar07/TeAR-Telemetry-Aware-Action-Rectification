# TeAR on SO-101

This directory contains the standalone five-joint SO-101 adapter runtime and a
servo-telemetry/control-loop template. Historical `tam_*` filenames remain usable.
For imports from an installed checkout, the preferred name is:

```python
from so101.tam_deploy.tam_runtime import SO101TAM

adapter = SO101TAM(checkpoint="weights/sft_adapter.pt", device="cpu")
adapter.assert_cool_identity()
```

The runtime accepts a six-component normalized action (five arm joints and a
gripper), five temperature/current/voltage entries, and the checkpoint's state
vector. It corrects the arm and passes the gripper through. These weights are
**not interchangeable with the seven-coordinate Panda OSC adapter**.

## Files

| File | Purpose |
|---|---|
| `tam_runtime.py` | `SO101TAM` standalone runtime |
| `tam_bot.py` | Vendored five-joint-compatible adapter for standalone deployment |
| `dynamixel_telemetry.py` | Historical filename for the Feetech serial-bus reader |
| `example_loop.py` | 30 Hz loop template with policy and motor-command hooks |

No weights are bundled. Supply a compatible checkpoint and configure dimensions
and architecture arguments to match its training configuration. For a standalone
copy of this directory, `from tam_runtime import SO101TAM` remains supported.

## Integration

1. Install PyTorch, NumPy, and the Feetech `scservo_sdk` in the robot environment.
2. Set the servo port, IDs, rated values, and register interpretation for your
   hardware in the telemetry reader. Verify the reported units against your
   servo documentation. Temperature is in °C; current and voltage supplied to
   the adapter are normalized by their rated values.
3. Load the checkpoint and run `assert_cool_identity()` without moving hardware.
4. Replace `BASE_POLICY` and the motor-command hook in `example_loop.py` with your
   actual interfaces. The template does not supply a policy or robot controller.

The nominal gate is closed at `T <= 42`, `C <= .60`, `V >= .90`. Exact pass-through
applies to finite deterministic network outputs and normalized input actions.
The script's target loop rate is configuration, not a measured hardware benchmark.

## Reported physical study

The manuscript reports Lift with a simulation-trained BC policy and TeAR, without
on-robot fine-tuning. There are 20 trials per method and condition: cool operation
below 45°C, shoulder heating above 50°C, and shoulder plus wrist heating above
50°C. Heating was induced by operating the arm under load. Success is 100%/100%,
75%/85%, and 70%/85% for base/TeAR, respectively. The study tests thermal transfer
on one task, not a physical current/voltage sweep. The experimental label “cool”
is not itself proof that every measured channel satisfies the stricter gate
thresholds.

"""Minimal real-time control loop for SO-101 + TAM.

This is a template. Wire your own base policy (π0 / GR00T / SmolVLA / ACT)
into the `BASE_POLICY` call site. Everything else stays the same regardless
of which policy you use.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dynamixel_telemetry import DynamixelTelemetry, fake_cool, fake_hot
from tam_runtime import SO101TAM


def main():
    # ---- TAM ----
    tam = SO101TAM(
        checkpoint=os.path.join(_HERE, "weights/sft_adapter.pt"),
        device="cuda",
    )
    tam.assert_cool_identity()
    print("[TAM] cool-identity verified", flush=True)

    # ---- Telemetry (real bus or fake for plumbing test) ----
    if os.environ.get("SO101_FAKE_TELEM", "0") == "1":
        print("[TELEM] using fake_hot() — TAM correction path will fire",
              flush=True)
        read_TCV = lambda: fake_hot()
    else:
        port = os.environ.get("SO101_PORT", "/dev/ttyACM0")
        telem = DynamixelTelemetry(port=port)
        read_TCV = telem.read_TCV
        print(f"[TELEM] connected to {port}", flush=True)

    # ---- Base policy ----
    # Replace this with your trained policy's inference call. The only
    # contract: it must produce a 6-DoF action in [-1, 1].
    def BASE_POLICY(obs):
        # placeholder: zero action with the gripper open
        return np.zeros(6, dtype=np.float32)

    # ---- Control loop ----
    obs = {"state": np.zeros(29, dtype=np.float32)}  # replace with real obs
    HZ = 30
    DT = 1.0 / HZ
    print(f"[LOOP] running at {HZ} Hz — Ctrl-C to stop", flush=True)
    try:
        while True:
            t0 = time.time()
            a_base = BASE_POLICY(obs)
            T, C, V = read_TCV()
            a_final = tam.correct(a_base, T, C, V, state=obs["state"])
            # send_to_motors(a_final)
            dt = time.time() - t0
            if dt < DT:
                time.sleep(DT - dt)
            else:
                print(f"[LOOP] overran {dt*1000:.1f}ms (>{DT*1000:.0f}ms target)",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[LOOP] stopped by user", flush=True)


if __name__ == "__main__":
    main()

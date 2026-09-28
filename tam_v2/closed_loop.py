"""Experimental confidence gain using simulator command-response feedback.

The evaluator supplies the synthetic post-degradation command, which is
privileged simulator information. A hardware observation estimator is not
provided. Scaling correction magnitude does not guarantee closed-loop safety.
This diagnostic is separate from the reference TeAR method.
"""

import numpy as np


class ClosedLoopGain:
    """Per-joint confidence gain on an adapter's proposed correction."""

    def __init__(self, n_joints=7, alpha_ema=0.2, mode="ratio",
                 rho_assumed=0.6, warmup=5, k_floor=0.0, eps=0.05):
        self.n = n_joints
        self.alpha = alpha_ema
        self.mode = mode
        self.rho_assumed = rho_assumed
        self.warmup = warmup
        self.k_floor = k_floor
        self.eps = eps
        self.reset()

    def reset(self):
        """Fresh deployment: assume healthy until measured otherwise."""
        self.rho_hat = np.ones(self.n, dtype=np.float32)
        self.steps = 0

    def update(self, a_commanded, a_executed):
        """EMA of the observed per-joint capacity. Mirrors
        _update_rho_estimate_from_obs in scripts/eval_transfer_any_base.py."""
        a_c = np.asarray(a_commanded, dtype=np.float32)[:self.n]
        a_e = np.asarray(a_executed, dtype=np.float32)[:self.n]
        mask = np.abs(a_c) > self.eps
        ratio = np.where(mask, a_e / (a_c + np.sign(a_c) * 1e-6), 1.0)
        ratio = np.clip(np.abs(ratio), 0.05, 1.5).astype(np.float32)
        # Only joints that actually moved carry information; hold the rest.
        self.rho_hat = np.where(
            mask, self.alpha * ratio + (1 - self.alpha) * self.rho_hat, self.rho_hat
        ).astype(np.float32)
        self.steps += 1

    def gain(self):
        """Per-joint k in [k_floor, 1]."""
        if self.steps < self.warmup:
            return np.ones(self.n, dtype=np.float32)
        measured_loss = np.clip(1.0 - self.rho_hat, 0.0, 1.0)
        if self.mode == "ratio":
            denom = max(1.0 - self.rho_assumed, 1e-3)
            k = measured_loss / denom
        elif self.mode == "linear":
            k = measured_loss
        else:
            raise ValueError(f"unknown mode {self.mode}")
        return np.clip(k, self.k_floor, 1.0).astype(np.float32)

    def apply(self, a_base, a_adapter):
        """Rescale the adapter's proposed correction by measured confidence."""
        a_b = np.asarray(a_base, dtype=np.float32)
        a_f = np.asarray(a_adapter, dtype=np.float32)
        out = a_f.copy()
        k = self.gain()
        m = min(self.n, len(a_b), len(a_f))
        out[:m] = a_b[:m] + k[:m] * (a_f[:m] - a_b[:m])
        return np.clip(out, -1.0, 1.0)


def selftest():
    """The two behaviours the design turns on."""
    g = ClosedLoopGain(warmup=0)
    a_b = np.array([0.5] * 7, np.float32)
    a_f = np.array([0.8] * 7, np.float32)          # adapter wants +0.3

    # Healthy arm: commanded == executed, so no authority was lost.
    for _ in range(30):
        g.update(a_b, a_b)
    healthy = g.apply(a_b, a_f)

    # Degraded arm: only 40% of the command is realised.
    g.reset()
    for _ in range(30):
        g.update(a_b, a_b * 0.4)
    degraded = g.apply(a_b, a_f)

    print(f"  healthy  rho_hat={g and 1.0:.2f}  ->  a_out[0]={healthy[0]:.3f}  "
          f"(a_base={a_b[0]:.2f}, adapter wanted {a_f[0]:.2f})")
    print(f"  degraded rho_hat=0.40  ->  a_out[0]={degraded[0]:.3f}")
    assert abs(healthy[0] - a_b[0]) < 1e-5, "correction must be suppressed when healthy"
    assert degraded[0] > healthy[0], "correction must pass through when degraded"
    print("  OK: suppressed when the arm is coping, applied when it is not.")


if __name__ == "__main__":
    print("ClosedLoopGain self-test")
    selftest()

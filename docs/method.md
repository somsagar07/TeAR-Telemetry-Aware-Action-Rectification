# Method and implementation

TeAR sits between a frozen policy and its controller. It receives the proposed
normalized action, a robot-state vector, and per-index temperature/current/voltage
telemetry. Seven action–telemetry tokens and a state token are processed by a
three-layer Transformer (width 128, four heads, feed-forward width 256, GELU,
pre-layer normalization, no dropout). Cartesian action indices and telemetry
indices are paired as an input representation, not a physical joint mapping.

## Correction and nominal identity

For each index, the smoothstep gate combines channel severities:

```text
zT = clip((T - 42) / 13, 0, 1)
zC = clip((C - .60) / .40, 0, 1)
zV = clip((.90 - V) / .40, 0, 1)
h(z) = z²(3 - 2z)
g = min(1, h(zT) + h(zC) + h(zV))

gamma = 1 + g * gamma_range * tanh(gamma_logit)
delta = g * alpha * tanh(delta_logit)
a_final = clip(gamma * a_base + delta, -1, 1)
```

With finite outputs and a normalized base action, `g = 0` gives exact action
identity. For the reference gain formula, correction magnitude is bounded by
`g * (gamma_range * abs(a_base) + alpha)`. These are command-level properties;
they do not establish closed-loop stability under stress. The experimental
log-space gain is a different parameterization and must be loaded as such.

## Simulation degradation

Capacity curves are **linear**, unlike the smoothstep gate:

| Channel | Nominal boundary | Full-degradation boundary | Capacity floor |
|---|---:|---:|---:|
| Temperature (°C) | 43 | 75 | 0.05 |
| Current (normalized) | 0.60 | 1.00 | 0.10 |
| Voltage (normalized) | 0.90 | 0.50 | 0.10 |

For each channel, joint capacities are averaged over zero-based joint groups
`[0,1,2,3]` for translation and `[3,4,5,6]` for rotation. Products of those
channel means scale the three translation and three rotation commands.
The simulator leaves the gripper unchanged.

Channels are applied sequentially (temperature, current, voltage), with clipping
after each. Reference noise standard deviations are `0.15*(1-rho)` for temperature,
`0.12*(1-rho)` for current, and `0.10*(1-rho)` for voltage, using the grouped
capacity. Current also adds `0.10*(1-rho)*sin(0.5*n + 2.094*i)` for coordinate
`i` and channel step `n`. Voltage noise has no autoregressive filter. The paired
protocol resets current phase per episode and isolates disturbance randomness.

## Supervised targets

SFT supplies demonstration actions as inputs and targets
`clip(a_demo / max(rhoT*rhoC*rhoV, .05), -1, 1)` at each of the six arm indices.
The gripper target remains the demonstration command. These index-wise labels
approximate the grouped evaluator and omit random disturbances. The denominator
floor is a training choice, not a floor on the environment's composed capacity.

| Tier | Temperature (°C) | Current | Voltage |
|---|---|---|---|
| Clean | 20–42 | .10–.50 | .92–1.00 |
| Mild | 43–55 | .50–.75 | .80–.92 |
| Moderate | 50–65 | .65–.90 | .65–.85 |
| Severe | 60–75 | .80–1.05 | .45–.70 |

Each demonstration time step receives one profile from each tier. The objective
is mean L1 target error plus three times the clean-sample identity loss. Training
uses AdamW (`lr=5e-4`, weight decay `1e-4`, default betas), cosine decay to zero,
no warm-up, and gradient-norm clipping at 1.0. Batch size counts augmented
state–action samples, not trajectories.

## Baseline information

- **Assumed inverse:** nominal linear curves, grouped capacities, clipped inverse;
  no actual test-curve information.
- **Privileged capacity reference:** actual test capacities, without disturbance
  cancellation; not an upper bound on task success.
- **MRAC, online SysID, experimental closed-loop gain:** the historical evaluator
  supplies the synthetic post-degradation command as response feedback. This is
  privileged simulator information, not a provided hardware estimator.

Both learned and analytic compensation depend on a telemetry-to-response prior.
The mismatch protocol measures transfer when that prior differs from the test
model. DR-TeAR is a separate trained variant; it is not the reference method.

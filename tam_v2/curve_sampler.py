"""Sample capacity-curve triples for randomized training and evaluation.

The reference TeAR uses linear training curves. Fresh draws from these families
vary parameters and channel combinations; they do not hold out whole families.
"""

import numpy as np

# Families available in env/thermal_model.py's CURVE_REGISTRY.
FAMILIES = ["linear", "exponential", "sigmoid", "polynomial", "kinky", "sin"]


def sample_channel(rng, ch, families=None, wide=True):
    """Draw one random monotone-ish capacity curve for channel ch in {T, C, V}.

    `wide=True` widens the rebuttal script's ranges, most importantly upward on
    `min_cap` (a higher floor = a milder curve, the regime TAM over-corrects in)
    and outward on onset, so training spans gentle through harsh degradation.
    """
    fams = families or FAMILIES
    fam = str(rng.choice(fams))

    if ch == "T":
        mc = float(rng.uniform(0.03, 0.45 if wide else 0.15))
        if fam == "linear":
            p = {"onset": float(rng.uniform(38, 55)), "full": float(rng.uniform(62, 92)), "min_cap": mc}
        elif fam == "exponential":
            p = {"onset": float(rng.uniform(38, 55)), "k": float(rng.uniform(0.02, 0.18)), "min_cap": mc}
        elif fam == "sigmoid":
            p = {"midpoint": float(rng.uniform(48, 70)), "k": float(rng.uniform(0.08, 0.40)), "min_cap": mc}
        elif fam == "polynomial":
            p = {"onset": float(rng.uniform(38, 55)), "full": float(rng.uniform(62, 92)),
                 "degree": int(rng.choice([2, 3])), "min_cap": mc}
        elif fam == "sin":
            p = {"onset": float(rng.uniform(38, 55)), "full": float(rng.uniform(62, 92)), "min_cap": mc}
        else:  # kinky -- piecewise, params fixed in the registry
            p = {"min_cap": mc}

    elif ch == "C":
        mc = float(rng.uniform(0.05, 0.50 if wide else 0.18))
        if fam == "linear":
            p = {"onset": float(rng.uniform(0.45, 0.78)), "full": float(rng.uniform(0.90, 1.20)), "min_cap": mc}
        elif fam == "exponential":
            p = {"onset": float(rng.uniform(0.45, 0.78)), "k": float(rng.uniform(2.0, 9.0)), "min_cap": mc}
        elif fam == "sigmoid":
            p = {"midpoint": float(rng.uniform(0.66, 0.96)), "k": float(rng.uniform(6.0, 18.0)), "min_cap": mc}
        elif fam == "polynomial":
            p = {"onset": float(rng.uniform(0.45, 0.78)), "full": float(rng.uniform(0.90, 1.20)),
                 "degree": int(rng.choice([2, 3])), "min_cap": mc}
        elif fam == "sin":
            p = {"onset": float(rng.uniform(0.45, 0.78)), "full": float(rng.uniform(0.90, 1.20)), "min_cap": mc}
        else:
            p = {"min_cap": mc}

    else:  # V -- degrades as voltage FALLS, so onset > full
        mc = float(rng.uniform(0.05, 0.50 if wide else 0.18))
        if fam == "linear":
            p = {"onset": float(rng.uniform(0.84, 0.98)), "full": float(rng.uniform(0.35, 0.65)), "min_cap": mc}
        elif fam == "exponential":
            p = {"onset": float(rng.uniform(0.84, 0.98)), "k": float(rng.uniform(2.0, 9.0)), "min_cap": mc}
        elif fam == "sigmoid":
            p = {"midpoint": float(rng.uniform(0.58, 0.82)), "k": float(rng.uniform(6.0, 18.0)), "min_cap": mc}
        elif fam == "polynomial":
            p = {"onset": float(rng.uniform(0.84, 0.98)), "full": float(rng.uniform(0.35, 0.65)),
                 "degree": int(rng.choice([2, 3])), "min_cap": mc}
        elif fam == "sin":
            p = {"onset": float(rng.uniform(0.84, 0.98)), "full": float(rng.uniform(0.35, 0.65)), "min_cap": mc}
        else:
            p = {"min_cap": mc}

    return fam, p


def sample_curve_set(n_curves, seed=0, families=None, wide=True):
    """n_curves independent (T, C, V) curve triples, as {ch: (family, params)}."""
    rng = np.random.default_rng(seed)
    return [{ch: sample_channel(rng, ch, families, wide) for ch in ("T", "C", "V")}
            for _ in range(n_curves)]


def describe(curves, k=5):
    lines = [f"{len(curves)} sampled curve triples (showing {min(k, len(curves))}):"]
    for i, c in enumerate(curves[:k]):
        lines.append(f"  [{i:02d}] " + "  ".join(
            f"{ch}={c[ch][0]}(min_cap={c[ch][1].get('min_cap', 0):.2f})" for ch in ("T", "C", "V")))
    fam_counts = {}
    for c in curves:
        for ch in ("T", "C", "V"):
            fam_counts[c[ch][0]] = fam_counts.get(c[ch][0], 0) + 1
    lines.append(f"  family mix: {dict(sorted(fam_counts.items()))}")
    return "\n".join(lines)


if __name__ == "__main__":
    cs = sample_curve_set(64, seed=0)
    print(describe(cs, k=6))

# H7 Analysis: Online Teleport — check_freq & lt_weight diagnosis

## Results Summary (as of 2026-04-16)

| Exp | check_freq | threshold | lt_weight | Class-IL | Δ vs baseline | triggers | avg_lt |
|-----|-----------|-----------|-----------|----------|---------------|----------|--------|
| E0-10ep | — | — | — | **58.53%** | baseline | 0 | — |
| E0-50ep | — | — | — | pending | — | 0 | — |
| E3 | 10 | 0.0 | 1.0 | **56.25%** | -2.28% | 850 | 5.67 |
| E2 | 1 | -0.3 | 1.0 | **48.87%** | -9.66% | 1834 | 4.80 |
| E1 | 1 | 0.0 | 1.0 | pending | — | 1113+ | TBD |

## Critical Finding: Loss Invariance is Broken

The teleport mechanism is **successfully improving gradient alignment** (cos_sim improves in ~99%
of trigger events), but at the cost of **massive loss disruption**:

- E3: avg_lt = 5.67, 87% of triggers have lt > 1.0
- E2: avg_lt = 4.80, 76% of triggers have lt > 1.0

With `lt_weight=1.0`, the objective `total_loss = 1.0*lt - cos_sim + reg*lora_reg` allows the
optimizer to sacrifice loss invariance for cos_sim gain. Since cos_sim typically improves by
~0.8–1.0, the optimizer willingly lets lt grow to 5–15.

### Impact on Training

Each teleport merges weights that change the loss by ~5.7 units. Normal training has loss ~1–2,
so this is catastrophic disruption. The subsequent gradient steps must "undo" this damage before
making useful progress. More frequent teleports (E2, freq=1) cause proportionally more damage,
explaining the larger accuracy drop (-9.66% vs -2.28%).

## Throughput Impact

| Exp | it/s (observed) | vs baseline (~50 it/s) | slowdown |
|-----|----------------|----------------------|----------|
| E3 (freq=10) | ~25 it/s | 50% throughput | 2x slower |
| E2 (freq=1, thr=-0.3) | ~11 it/s | 22% throughput | 4.5x slower |
| E1 (freq=1, thr=0.0) | ~4 it/s | 8% throughput | 12x slower |

Conclusion: `freq=1` with `threshold=0.0` is **impractical** (12x slowdown) even ignoring accuracy.

## Root Cause

`lt_weight=1.0` is grossly insufficient to enforce loss invariance. The optimizer greedily
maximizes cos_sim improvement while ignoring the L_t constraint.

**Fix**: Increase `lt_weight` substantially (e.g., 10, 50, 100) so that loss disruption is
heavily penalized. With high lt_weight, the optimizer should find LoRA perturbations that
improve gradient alignment without changing the loss value.

## Next Experiments (H7b)

| Exp | check_freq | threshold | lt_weight | n_steps | prediction |
|-----|-----------|-----------|-----------|---------|-----------|
| E4 | 10 | 0.0 | 10.0 | 10 | lt stays < 0.5, less disruption, maybe better than baseline |
| E5 | 10 | 0.0 | 50.0 | 10 | lt ≈ 0, almost no disruption, see if cos_sim still improves |
| E6 | 10 | 0.0 | 10.0 | 20 | more steps to converge with stronger constraint |

Key question: Can we find a lt_weight that:
1. Keeps lt small (< 0.1) — loss invariance holds
2. Still achieves cos_sim > 0 improvement
3. Doesn't slow training too much

## Lessons

1. **Never run freq=1 with threshold=0**: 12x slowdown, not viable
2. **lt_weight must be >> 1**: 1.0 effectively disables the loss invariance constraint
3. **The cos_sim improvement mechanism works** (99% success), but needs to be loss-neutral
4. **freq=10 is a reasonable tradeoff**: 2x slowdown is acceptable if accuracy improves

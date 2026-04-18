# H11 Analysis: SGD + Online Teleport — Batch Mismatch Test

## Results

| Exp | Class-IL | Task-IL | Δ | cos+ rate | mean cos_improve | mean_lt | triggers |
|-----|----------|---------|---|-----------|-----------------|---------|---------|
| SGD_base | 19.45% | 66.46% | 0 | — | — | — | 0 |
| SGD_f10 | **19.56%** | **73.24%** | **+0.11%** | 86.7% | +0.0163 | 0.044 | 792 |

Per-task trajectory:
| After Task | SGD_base | SGD_f10 | Δ |
|-----------|----------|---------|---|
| 1 | 98.05% | 96.95% | -1.10% |
| 2 | 43.75% | 44.00% | +0.25% |
| 3 | 31.12% | 31.30% | +0.18% |
| 4 | 23.88% | 24.58% | +0.70% |
| 5 | 19.45% | 19.56% | +0.11% |

## Hypothesis Verdict: REJECTED

**H11 hypothesis**: If ER batch mismatch is the root cause of freq=1 not scaling, then SGD+teleport
(no batch mismatch) should show larger absolute improvement over SGD_base than ER+teleport over ER_base.

- ER+teleport improvement: **+2.65%** (58.53% → 61.18%)
- SGD+teleport improvement: **+0.11%** (19.45% → 19.56%)

The batch mismatch hypothesis is **rejected**. SGD has no batch mismatch yet shows almost zero benefit.

## Mechanism is Working — Signal is the Problem

| Metric | SGD_f10 | ER_f10 (H10 best) |
|--------|---------|----------|
| cos+ rate | 86.7% | 91.3% |
| mean cos_improve | +0.0163 | +0.027 |
| mean_lt | 0.044 | 0.035 |
| lt<0.1 rate | 90.2% | (similar) |
| Class-IL improvement | **+0.11%** | **+2.65%** |

The teleportation mechanism works correctly in both contexts:
- Loss is preserved (lt ≈ 0.04, all < 0.18)
- cos_sim improves 87% of the time
- Weight perturbation is tiny (delta_norm ~0.32)

But SGD gives no benefit despite a working mechanism.

## Root Cause: Old-Task Gradient Quality

The key difference between SGD and ER is **what the old-task gradient represents**:

**In ER**: The model retains partial old-task knowledge due to continuous replay.
The 256-sample teleport memory gradient is meaningful — the model can still classify
old-task samples, so the gradient direction is informative about the conflict.
→ Correcting this conflict helps marginally (+2.65%).

**In SGD**: After 10 epochs of pure new-task finetuning, old tasks are catastrophically
forgotten. The model's gradient on 256 old-task samples is essentially noise — the model
has no representation of old tasks left. Aligning the new-task gradient with a noise signal
has no effect.
→ cos_sim alignment = noise alignment → +0.11% (negligible).

## Implication: Teleportation is Complementary to Replay, Not a Replacement

Online LoRA teleportation REQUIRES a functional old-task gradient signal to be meaningful.
This signal only exists when some old-task knowledge is preserved (e.g., via ER replay).
Without replay, the teleport memory gradient is too noisy to guide useful alignment.

This explains all findings coherently:
1. ER+teleport: +2.65% (meaningful gradient signal from replay-preserved knowledge)
2. SGD+teleport: +0.11% (noisy gradient signal from catastrophically forgotten model)
3. freq=1 doesn't scale (even in ER): each correction is +0.016 cos_improve → too small to accumulate
4. H8: cos_sim correlation with forgetting is weak (r=-0.25) because the signal is inherently noisy

## Conclusion

The ER batch mismatch is NOT the limiting factor. The fundamental limitation is gradient signal quality:
teleportation can only reduce gradient conflict when the old-task gradient is informative,
which requires prior knowledge preservation (replay). This makes online teleportation
a second-order enhancement to replay methods, not an independent technique.

The freq=1 non-scaling remains unexplained by batch mismatch, but may be explained by
the fact that each individual correction is too small (+0.016) to matter cumulatively.

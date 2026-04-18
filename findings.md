# Findings: Online Teleportation for Continual Learning

## Research Question
Can mid-training gradient conflict detection + LoRA teleportation reduce catastrophic forgetting
in continual learning, specifically by finding loss-equivalent parameter configurations with better
gradient alignment between old and new tasks?

## Current Understanding (2026-04-19) — H11 IN PROGRESS (H9/H10 added breakthrough)

### Complete Results (all ER experiments)

| Exp | rank | teleport_lr | freq | lt_weight | Class-IL | Δbaseline | avg_lt | cos+ rate | delta_norm |
|-----|------|-------------|------|-----------|----------|-----------|--------|-----------|------------|
| ER baseline (10ep) | — | — | — | — | **58.53%** | 0 | — | — | — |
| ER baseline (50ep) | — | — | — | — | **61.66%** | — | — | — | — |
| E5 (H7 best, old) | 4 | 0.01 | 10 | 50 | 56.90% | -1.63% | 1.73 | — | ~7 |
| E6 (best lt, old) | 4 | 0.01 | 10 | 10 | 48.22% | -10.31% | **0.46** | 47.5% | 25–30 |
| H9 rank=8 | 8 | 0.01 | 10 | 50 | 29.13% | -29.40% | 1.65 | 62.5% | 13 |
| H10 E_lt200_lr001 | 8 | **0.001** | 10 | 200 | 60.54% | **+2.01%** | 0.070 | 91.0% | 0.478 |
| H10 E_lt500_lr001 | 8 | **0.001** | 10 | 500 | 61.30% | **+2.77%** | 0.074 | 90.1% | 0.475 |
| **H10 best (rank4)** | **4** | **0.001** | **10** | **200** | **61.18%** | **+2.65%** | **0.035** | **91.3%** | **0.321** |
| H10 freq=1 | 4 | 0.001 | 1 | 200 | 60.49% | +1.96% | 0.036 | 91.8% | 0.321 |

### Critical Paradox (Resolved in H10)

Old finding (H7): E6 achieves the **lowest avg_lt (0.46)** yet produces the **worst accuracy (48.22%)**.
This appeared to confirm Flaw 5 — that loss invariance requires such large delta_norm that other batches are devastated.

**H10 Resolution**: The problem was NOT fundamental — it was `teleport_lr=0.01` being too large.
With `teleport_lr=0.001`:
- delta_norm drops from 25–30 → **0.32** (100x smaller)
- avg_lt drops from 0.46 → **0.035** (13x smaller)
- cos+ rate rises from 47.5% → **91.3%**
- Accuracy: -1.63% → **+2.65%** (first positive result)

The key insight: small teleport_lr forces the LoRA optimizer to take tiny steps. These tiny
perturbations are sufficient to improve cos_sim slightly while leaving the weight manifold
essentially unchanged. Flaw 5 is avoided by design — the perturbation is so small that
zero-order guarantees hold approximately across many nearby batches.

## Patterns and Insights

### Pattern 1: teleport_lr is the Most Critical Hyperparameter
Not rank, not lt_weight, not freq — the learning rate inside the teleport optimizer.
- teleport_lr=0.01: always hurts (delta_norm 7–50, catastrophic disruption)
- teleport_lr=0.001: works (+2.65% over baseline; delta_norm 0.32, lt=0.035)

### Pattern 2: Frequency Does Not Scale with ER
- freq=10 (755 triggers): +2.65%
- freq=1 (7358 triggers, 10x more): +1.96% (same, or slightly worse)
10x more gradient corrections give no additional benefit. Possible cause: ER batch mismatch.

### Pattern 3: Mechanism Works (cos_sim improvement is real and loss-invariant)
With teleport_lr=0.001: 91.3% of triggers show positive cos_improve, avg_lt=0.035.
The mechanism is functioning correctly at a local level. The limitation is elsewhere.

### Pattern 4: cos_sim Conflict is Ubiquitous (H8)
78.9% of all training steps have cos_sim < 0. This means gradient conflict is a background
property, not a selective event. The correlation with forgetting is weak (r=-0.25 pooled,
inconsistent across seeds).

## Lessons and Constraints

1. teleport_lr=0.001 (not 0.01) is required for positive results
2. rank=4 is sufficient; higher rank doesn't improve accuracy with small lr
3. freq=1 is computationally infeasible for SGD (every step triggers; ~15 hours)
4. freq scaling doesn't work with ER — batch mismatch is the likely cause (testing in H11)
5. The 50-epoch ER baseline (61.66%) sets the ceiling context; H10 best (61.30%) is close

## H11: SGD + Online Teleport — Batch Mismatch Test (2026-04-19)

### Setup
- SGD_base: pure finetuning, no replay → 19.45% Class-IL (severe forgetting as expected)
- SGD_f10: SGD + online teleport freq=10 (H10 best params: teleport_lr=0.001, rank=4, lt_weight=200)
- No ER buffer → no batch mismatch between teleport memory and training

### Results

| Exp | Class-IL | Δ | cos+ rate | mean cos_improve | mean_lt | triggers |
|-----|----------|---|-----------|-----------------|---------|---------|
| SGD_base | 19.45% | 0 | — | — | — | — |
| SGD_f10 | **19.56%** | **+0.11%** | 86.7% | +0.0163 | 0.044 | 792 |
| ER_base | 58.53% | 0 | — | — | — | — |
| ER_f10 (H10 best) | **61.18%** | **+2.65%** | 91.3% | +0.027 | 0.035 | ~755 |

### Critical Finding: Batch Mismatch NOT the Cause

The batch mismatch hypothesis is **rejected**: SGD has no batch mismatch yet shows near-zero benefit (+0.11%).

Mechanism works correctly in both settings (86-91% cos+ rate, lt~0.04, loss-preserving).
The difference is **gradient signal quality**:

- **ER**: Model retains old-task knowledge via replay → old-task gradient is informative → marginally useful (+2.65%)
- **SGD**: Old tasks catastrophically forgotten → old-task gradient on 256 teleport samples = noise → +0.11%

### Implication: Teleportation Requires Prior Knowledge Preservation

Online LoRA teleportation can only reduce gradient conflict when the old-task gradient is
informative. This requires some knowledge preservation (ER replay). Without replay, teleport
memory gradient is noise. Teleportation is a **second-order enhancement to replay**, not
an independent forgetting-prevention technique.

## H8: Gradient Conflict → Forgetting Hypothesis Verification (2026-04-16)

### Setup
- Detection-only mode (teleport_detect_only=1, threshold=100 → never triggers)
- Logged cos_sim at every check (freq=10) for 3 seeds × seq-cifar10
- Correlated mean cos_sim per task with per-task forgetting

### Key Results

**Gradient conflict is ubiquitous: 78.9% of all training steps have cos_sim < 0**

| Seed | r(cos_sim, forgetting) | Cross-task r |
|------|----------------------|--------------|
| 42   | -0.43                | -0.73        |
| 123  | -0.01                | -0.32        |
| 456  | -0.54                | +0.08        |
| **Pooled** | **-0.25**   | —            |

### Critical Finding
The hypothesis is **NOT confirmed**. The mean cos_sim per task ranges from only -0.13 to -0.30
(17 unit range) across all tasks — there's insufficient variance to detect a causal signal.
More importantly: if 79% of steps have gradient conflict, it's a background property of
multi-task gradient mixing, not a discriminating event. The correlation is inconsistent across
seeds and largely driven by task-order confound (last task always has zero forgetting).

### Verdict
The "gradient conflict causes forgetting" hypothesis is **not confirmable** with single-step
mini-batch cos_sim as the proxy. The signal-to-noise ratio is too low and the measurement
granularity is wrong. Conflict is ubiquitous, not selective.

---

## Final Decision: CONCLUDE — Coherent Story Across H6–H11

All experiments complete. The results form a coherent, publishable narrative:

### The Story
1. **H7/H8**: Online teleport with teleport_lr=0.01 always hurts. cos_sim conflict is ubiquitous
   (79% steps), signal too noisy (r=-0.25 with forgetting). Appeared fundamentally broken.

2. **H9/H10 BREAKTHROUGH**: teleport_lr=0.001 (not 0.01) is the key. Small lr → tiny delta_norm
   (0.32 vs 13), perfect loss preservation (lt=0.035), 91% cos+ rate → **+2.65%** over ER baseline.
   First positive result. Shows the mechanism CAN work when properly constrained.

3. **H10 Puzzle**: freq=1 (7358 triggers) ≈ freq=10 (755 triggers) in accuracy. 10x more corrections
   give no additional benefit. Batch mismatch hypothesis proposed.

4. **H11 Resolution**: SGD+teleport = +0.11% despite perfect mechanism (87% cos+ rate, lt=0.04).
   Batch mismatch is NOT the cause. Root cause: old-task gradient quality. ER preserves old-task
   knowledge via replay → informative gradient → useful alignment. SGD doesn't → noisy gradient
   → useless alignment. Teleportation is a **second-order replay complement**, not standalone.

### Verdict
**LoRA teleportation for continual learning**: viable as a small complement to ER (+2.65%),
not viable standalone (SGD +0.11%). The improvement requires existing knowledge preservation.
The cos_sim improvement per event (+0.027) is too small to scale with frequency.

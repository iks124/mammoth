# H8 Analysis: Gradient Conflict → Forgetting Hypothesis

## Results (2026-04-16)

### cos_sim distribution during training (ER, seq-cifar10, 10 epochs)

| Seed | Task 1 | Task 2 | Task 3 | Task 4 | Mean neg% |
|------|--------|--------|--------|--------|-----------|
| 42   | -0.173 (70.6%) | -0.252 (83.1%) | -0.214 (77.0%) | -0.133 (75.1%) | 76.5% |
| 123  | -0.179 (73.2%) | -0.216 (78.6%) | -0.304 (89.1%) | -0.155 (78.3%) | 79.8% |
| 456  | -0.234 (80.8%) | -0.221 (79.6%) | -0.268 (84.7%) | -0.131 (76.4%) | 80.4% |
| **Mean** | **-0.195** | **-0.230** | **-0.262** | **-0.140** | **78.9%** |

*Values: mean cos_sim (% steps with cos_sim < 0)*

### Per-task forgetting (ER baseline, 10 epochs)

| Seed | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 |
|------|--------|--------|--------|--------|--------|
| 42   | +50.85% | +53.95% | +37.20% | +20.45% | 0% |
| 123  | +50.00% | +60.00% | +42.70% | +22.15% | 0% |
| 456  | +53.40% | +42.35% | -1.50%  | +20.30% | 0% |

### Correlation: cos_sim during task t vs forgetting of task t

| Seed | r (direct) | r (cross: cos_t vs forget_{t-1}) |
|------|------------|----------------------------------|
| 42   | -0.4345    | -0.7304 |
| 123  | -0.0097    | -0.3193 |
| 456  | -0.5393    | +0.0813 |
| **Pooled (12 pts)** | **-0.2485** | — |
| **Without last task (9 pts)** | **+0.5245** | — |

## Critical Findings

### Finding 1: Gradient Conflict is Ubiquitous

**78.9% of training steps have cos_sim(g_old, g_new) < 0.**

This is the single most important result. The hypothesis was:
> "Gradient conflict causes forgetting, so reducing it should reduce forgetting."

But if conflict occurs on 79% of ALL training steps, it cannot be a selective signal for
"this batch will cause forgetting." It's background noise throughout training.

This explains why H1-H7 teleportation never worked: reducing conflict on a random 0.5-5% of
steps (the ones the optimizer happens to "fix") cannot systematically affect the other 79% of
conflicting steps.

### Finding 2: No Consistent Correlation

- Mean direct r = -0.33 (weak, high variance across seeds: -0.43, -0.01, -0.54)
- The pooled r = -0.25 is largely driven by the trivial task-order effect:
  - Task 4 (last): always forgetting=0, cos_sim least negative (-0.14)
  - Task 1 (early): most forgetting, cos_sim moderately negative (-0.20)
  - This pattern reflects TASK ORDER, not a causal cos_sim → forgetting path

- Without the last task (where forgetting=0 by definition), r = +0.5245,
  which means HIGHER conflict is weakly associated with LESS forgetting.
  This is the OPPOSITE of the hypothesis (but unreliable with only 9 data points).

### Finding 3: The Mean cos_sim Has Too Little Variance

Across all tasks and seeds, mean cos_sim ranges from **-0.13 to -0.30** — a narrow band of
0.17 units. The measurement lacks statistical power to discriminate causal effects.

## Verdict: Hypothesis NOT Confirmed

The gradient conflict → forgetting hypothesis is **not supported** with this measurement setup.

**Not because gradient alignment doesn't matter** — it likely does, at the parameter space level.
But cos_sim of last-layer gradients on a single mini-batch is too noisy a proxy.

Possible interpretation: conflict is ubiquitous during all multi-task gradient mixing. The
relevant factor is not whether conflict happens, but how the optimizer accumulates these
conflicting updates over 3130 steps. A single-step cos_sim has no predictive power over
this trajectory.

## What This Means for the Research Programme

1. H6/H7: Online teleport doesn't work (Flaw 5 + ubiquitous conflict)
2. H8: The "gradient conflict causes forgetting" hypothesis is not confirmable with this proxy
3. Teleportation designed around cos_sim is solving the wrong problem, or solving it
   at the wrong granularity

**The teleportation approach as designed is not viable.** This is a coherent negative result.

## Alternative Interpretations (not tested)

- Accumulated gradient conflict (cumulative cosine over trajectory) might correlate better
- Gradient conflict at specific "critical steps" (end of task) might matter more
- The signal might be cleaner with class-conditional gradients (within-task vs between-task)

These would require substantially different measurement designs.

# Findings: Online Teleportation for Continual Learning

## Research Question
Can mid-training gradient conflict detection + LoRA teleportation reduce catastrophic forgetting
in continual learning, specifically by finding loss-equivalent parameter configurations with better
gradient alignment between old and new tasks?

## Current Understanding (2026-04-16)

### What Works
- **Gradient conflict detection is reliable**: `detect_gradient_conflict` correctly identifies
  steps where old/new task gradients conflict (cos_sim < 0), confirmed by diagnostic runs.
- **cos_sim improvement mechanism works locally**: When lt_weight=1.0 (unconstrained),
  teleportation improves gradient alignment in **~99% of triggered events**, with average
  cos_sim improve of +0.8 to +1.0.

### What Doesn't Work (Yet)
- **All online teleport variants hurt accuracy** compared to ER baseline:
  - E0-10ep (ER baseline, 10ep): **58.53%**
  - E3 (freq=10, thr=0.0, lt=1.0): **56.25%** (-2.28%)
  - E2 (freq=1, thr=-0.3, lt=1.0): **48.87%** (-9.66%)

### Root Cause: Loss Invariance Failure
With `lt_weight=1.0`, the optimizer sacrifices loss invariance entirely:
- avg_lt = 5.67 (E3), 4.80 (E2) — loss changes by ~5 units per teleport
- 87% of teleports have lt > 1.0
- Each teleport is equivalent to a large noisy gradient step, disrupting training

**Physical interpretation**: The LoRA optimizer finds a direction that improves cos_sim, but it
takes a large step off the loss manifold. The merged weights then need many subsequent training
steps to "recover". More teleports = more disruption = worse accuracy.

### New Round (H7b): Higher lt_weight
Running E4 (lt=10), E5 (lt=50), E6 (lt=10, 20 steps). Early results:
- E6: lt constrained (0.38) but delta_norm huge (25-30) and cos_sim gets **worse** — 
  the optimizer moves along the loss-flat manifold but can't find better gradient alignment
- E4: inconsistent — sometimes lt=0.26 (good), sometimes lt=8.83 (unconstrained)
- E5: lt moderate (0.5-0.9), cos_sim modestly improved (+0.1 to +0.4)

## Patterns and Insights

### The Bias-Variance Tradeoff for lt_weight
- **Low lt_weight (1.0)**: cos_sim improves a lot BUT loss is destroyed → accuracy drops
- **High lt_weight (10-50)**: loss preserved BUT cos_sim barely improves, or gets worse
- No obvious sweet spot found yet

### Throughput is a Hard Constraint
- freq=1: 8-12% of baseline throughput (4-11 it/s vs ~50 it/s). **Not viable**.
- freq=10: ~50% throughput (25 it/s). **Acceptable** if accuracy improves.
- The second-order gradient computation (create_graph=True) dominates cost per teleport.

### The Core Tension
Online teleportation must find a point P such that:
1. L(P) ≈ L(θ) — loss invariant (lt ≈ 0)
2. cos_sim(g_old(P), g_new(P)) > cos_sim(g_old(θ), g_new(θ)) — gradient alignment improves
3. ‖P - θ‖ small — not too far from current weights

Conditions 1 and 2 may be **fundamentally incompatible** with rank-2 LoRA in the online setting:
the loss-flat subspace (condition 1) may not contain points with better gradient alignment
(condition 2), especially with only 5-20 optimization steps and a small rank.

## Lessons and Constraints

1. **Never use freq=1, threshold=0.0**: 12x slowdown, not viable
2. **lt_weight=1.0 is too weak**: effectively disables loss invariance, lt averages 5.7
3. **freq=10 is the right frequency**: 2x slowdown, acceptable if accuracy improves  
4. **Online teleport at task boundary (H6b)**: tried threshold=-0.3, gave -0.15% 
   (nearly neutral, very stable std=0.74%) — better than mid-training online
5. **LoRA rank=2 may be too small**: insufficient degrees of freedom to satisfy both constraints

## Open Questions

1. Does any lt_weight value achieve lt < 0.1 while still improving cos_sim?
2. Does the online approach work at all with the current rank-2 LoRA parameterization?
3. Should we pivot back to task-boundary teleportation (H6b) which was nearly neutral?
4. Is the fundamental hypothesis (gradient conflict → forgetting) verified empirically?
5. SAM-style first-order approximation: would it be faster AND more effective?

## Experiment Trajectory

| Run | Config | Class-IL | Δ baseline | Status |
|-----|--------|----------|-----------|--------|
| E0-10ep | ER 10ep | 58.53% | baseline | DONE |
| E0-50ep | ER 50ep | pending | — | running |
| E3 | freq=10, lt=1.0 | 56.25% | -2.28% | DONE |
| E2 | freq=1, thr=-0.3, lt=1.0 | 48.87% | -9.66% | DONE |
| E1 | freq=1, thr=0.0, lt=1.0 | pending | — | running (slow) |
| E4 | freq=10, lt=10 | pending | — | running |
| E5 | freq=10, lt=50 | pending | — | running |
| E6 | freq=10, lt=10, 20steps | pending | — | running |

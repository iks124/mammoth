# Findings: Online Teleportation for Continual Learning

## Research Question
Can mid-training gradient conflict detection + LoRA teleportation reduce catastrophic forgetting
in continual learning, specifically by finding loss-equivalent parameter configurations with better
gradient alignment between old and new tasks?

## Current Understanding (2026-04-16) — H7 COMPLETE

### Complete Results

| Exp | freq | lt_weight | steps | Class-IL | Δbaseline | triggers | avg_lt |
|-----|------|-----------|-------|----------|-----------|----------|--------|
| E0-10ep (baseline) | — | — | — | **58.53%** | 0 | 0 | — |
| E0-50ep (baseline) | — | — | — | **61.66%** | — | 0 | — |
| E5 (best) | 10 | 50 | 10 | 56.90% | -1.63% | 972 | 1.73 |
| E3 | 10 | 1.0 | 5 | 56.25% | -2.28% | 850 | 5.67 |
| E4 | 10 | 10 | 10 | 53.51% | -5.02% | 927 | 1.92 |
| E2 | 1 | 1.0 | 5 | 48.87% | -9.66% | 1834 | 4.80 |
| E6 (best lt!) | 10 | 10 | 20 | 48.22% | -10.31% | 1140 | **0.46** |
| E1 | 1 | 1.0 | 5 | 26.93% | -31.60% | 4419 | — |

### Critical Paradox: Better lt → Worse Accuracy

E6 achieves the **lowest avg_lt (0.46)** — loss invariance is actually working — yet produces
the **worst accuracy (48.22%)**.

Root cause: **Flaw 5 confirmed experimentally**. To achieve low lt on the current batch, the
optimizer must make a HUGE LoRA delta (delta_norm=25–30). This preserves loss on the specific
batch but catastrophically disrupts behavior on all other batches. The "zero-order guarantee"
(loss preserved at one point) does not cover the training trajectory.

### Two-Sided Trap

| Regime | lt | delta_norm | Effect |
|--------|----|-----------|--------|
| Unconstrained (lt_weight=1) | 5.67 | 4–6 | Current batch loss disrupted |
| Moderate (lt_weight=50) | 1.73 | 9–13 | Partial disruption |
| Strongly constrained (lt_weight=10, 20steps) | 0.46 | 25–30 | Other batches devastated |

There is **no viable middle ground** with rank-2 LoRA in 5–20 steps.

## Patterns and Insights

### Pattern 1: Online Teleport Consistently Hurts
All 6 online teleport variants are worse than baseline. The best result (E5: 56.90%) is still
-1.63% below 10-epoch baseline. This is not a hyperparameter issue — it's structural.

### Pattern 2: Frequency is Critical, But Not Sufficient
- freq=1: 12x slower, -31.6% accuracy (catastrophic)
- freq=10: 2x slower, -1.6% to -10.3% (bad to catastrophic)
Even optimal frequency doesn't overcome the core issue.

### Pattern 3: cos_sim Improvement is Not Enough
The teleport improves cos_sim in 93–99% of cases (mechanism works locally), but:
1. The improvement doesn't persist across steps
2. Each teleport creates a perturbation that hurts subsequent steps
3. Many teleports per task (850–4419) compound the damage

## Lessons and Constraints

1. Online teleport (H6/H7) is **fundamentally broken** with rank-2 LoRA
2. freq=1 is never viable (12x slowdown, catastrophic accuracy)
3. lt_weight tuning cannot fix the structural issue (Flaw 5)
4. delta_norm must be kept small to avoid disrupting other batches — but then lt can't be controlled
5. The 50-epoch baseline (61.66%) is the proper reference for the full method

## Open Questions / Next Directions

### Option A: PIVOT to task-boundary teleport improvement
H6b (task-boundary, threshold=-0.3): -0.15%, std=0.74% (neutral/stable)
Can we understand why H6b is neutral and push it positive?

### Option B: Verify the fundamental hypothesis
Does gradient conflict actually CAUSE forgetting? If not, teleportation is solving the wrong problem.
Test: measure correlation between per-step cos_sim and eventual forgetting on that task.

### Option C: SAM-style first-order approximation
Replace create_graph=True with weight perturbation:
g_approx = (g(θ + ε*sign(g)) - g(θ)) / ε
Much cheaper → can do more steps or higher rank without blowing up compute.

### Option D: Conditional merge
Only merge if both: (a) lt < 0.1 AND (b) cos_sim actually improved.
Filter out bad teleports. Roughly 5% of events would pass. Very few merges = little disruption.
But: is this enough to help? Unclear.

## Decision: PIVOT

Online teleport does not work. The theoretical flaws (Flaw 5: zero-order ≠ trajectory) are
confirmed experimentally. Recommend pivoting to either:
1. Investigating why H6b (task-boundary) is neutral
2. Verifying the gradient conflict → forgetting hypothesis itself

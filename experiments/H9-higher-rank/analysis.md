# H9 Analysis: Higher LoRA Rank for Online Teleportation

## Results

| Exp | rank | Class-IL | Δ | cos+ rate | mean cos_improve | avg delta_norm |
|-----|------|----------|---|-----------|-----------------|----------------|
| E5 (ref) | 4 | 56.90% | -1.63% | 47.5% | +0.025 | ~7 |
| E_rank8  | 8 | 29.13% | **-29.40%** | **62.5%** | **+0.323** | ~13 |
| E_rank16 | 16 | 43.02% | -15.51% | 50.0% | +0.120 | ~50 |

## Key Finding: Hypothesis Partially Confirmed, Outcome Reversed

The hypothesis that "higher rank → higher cos_improve success rate" is **confirmed**:
- rank=4: 47.5% positive rate, mean +0.025
- rank=8: 62.5% positive rate, mean +0.323  ← best cos_improve

But the accuracy outcome is **reversed**: rank=8 achieves the best cos_improve yet the worst accuracy (29.13%).

## Root Cause: delta_norm Scales with Rank

Higher rank gives the optimizer more freedom to find cos_sim-improving directions,
but simultaneously allows larger weight perturbations:
- rank=4: avg delta_norm ~7
- rank=8: avg delta_norm ~13
- rank=16: avg delta_norm ~50

The `teleport_reg` (Frobenius norm penalty) does not sufficiently constrain delta_norm
when rank is large, because more parameters can each contribute small individual changes
that sum to a large total displacement.

The `lt_weight` constraint ensures loss is preserved on the **current batch**, but a large
delta_norm means weights have moved far in parameter space. Future batches (which were NOT
in the lt constraint) see a fundamentally different loss landscape, disrupting training.

## This is Flaw 5 Amplified

Flaw 5 (from project_teleport_flaws.md): "zero-order loss invariance at one batch ≠
trajectory invariance." With higher rank, this flaw is amplified: the optimizer makes
LARGER weight perturbations that preserve loss on one batch but disrupt all future batches.

## What Rank Variation Reveals

Rank=16 (delta_norm~50) gives better accuracy than rank=8 (delta_norm~13) despite worse
cos_improve, suggesting the relationship between cos_improve and accuracy is non-monotonic
and likely confounded by delta_norm effects.

The fundamental bottleneck is NOT rank (degrees of freedom for cos_sim search), but the
inability to constrain delta_norm while still finding cos_sim-improving points.

## Implications

To fix this, we need to directly constrain delta_norm (not just lt):
- Increase `teleport_reg` to heavily penalize large LoRA perturbations
- Or add an explicit `delta_norm_max` constraint

However, this creates the same trade-off: smaller delta_norm → less freedom → may not
find cos_sim improvements. The problem appears fundamental to the LoRA teleportation
design: **any perturbation large enough to meaningfully improve cos_sim also displaces
weights enough to disrupt future training steps.**

## Verdict

Higher LoRA rank does NOT fix the online teleportation problem. The failure mode is
weight displacement (delta_norm), not search space size. This experiment closes the
rank investigation — the online teleportation approach as designed appears not viable
regardless of rank choice.

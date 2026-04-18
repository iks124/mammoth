# H10: rank=8 + lt_weight sweep — Can we enforce loss invariance while keeping cos_improve positive?

## Hypothesis

H9 (rank=8, lt_weight=50) showed: cos_improve positive rate=62.5%, mean cos_improve=+0.323,
but mean lt=1.651 and acc=29.13% (worse than baseline). The lt is still too large even at
lt_weight=50 because cos_sim gain still dominates the penalty.

With rank=8 we have enough degrees of freedom. If we increase lt_weight to 200–500,
the optimizer is forced to keep lt small. With rank=8 (vs rank=4 in E6 which turned
cos_improve negative under strict lt), there may be enough freedom to satisfy both constraints.

Two additional levers:
1. `lt_weight`: forces loss invariance directly
2. `teleport_reg`: Frobenius norm penalty on delta (limits delta_norm, keeping weights close)

**Prediction**: At lt_weight=200 or 500, rank=8 will have avg_lt < 0.3 AND cos+ rate > 50%.
If so, accuracy should improve beyond E5 (56.90%).

## Experiments (rank=8, all other params same as E5)

| ID              | lt_weight | teleport_reg | Expected effect            |
|-----------------|-----------|--------------|---------------------------|
| E_lt200         | 200       | 0.1 (same)   | lt↓, see if cos+ survives |
| E_lt500         | 500       | 0.1 (same)   | lt↓↓, test extreme lt constraint |
| E_lt200_reg1    | 200       | 1.0          | lt↓ + delta_norm↓ together |

## Fixed (same as E5/H9 rank=8 baseline)
- rank=8, freq=10, n_steps=10, threshold=0.0
- dataset: seq-cifar10, er, buffer_size=500, lr=0.1, n_epochs=10, seed=42

## Key Metrics
- avg_lt (want < 0.3)
- cos_improve positive rate (want > 50%)
- mean cos_improve (want > 0)
- avg delta_norm (want to see if it decreases)
- Class-IL accuracy

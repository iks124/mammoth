# H10 Analysis: rank=8 lt_weight sweep + teleport_lr discovery

## Key Result: teleport_lr=0.001 is the Critical Parameter

| ID | rank | lt_weight | teleport_lr | Class-IL | Δbaseline | avg_lt | cos+ rate | mean_cos_improve | delta_norm |
|----|------|-----------|-------------|----------|-----------|--------|-----------|-----------------|------------|
| H9_E_rank8 (ref) | 8 | 50 | 0.01 | 29.13% | -29.40% | 1.651 | 62.5% | +0.323 | 13.0 |
| E_lt200 | 8 | 200 | 0.01 | 47.17% | -11.36% | 1.428 | 89.7% | +0.536 | 15.68 |
| E_lt500 | 8 | 500 | 0.01 | 49.73% | -8.80% | 1.724 | 89.9% | +0.555 | 16.38 |
| E_lt200_reg1 | 8 | 200 | 0.01 | 42.23% | -16.30% | 2.043 | 89.6% | +0.541 | 12.23 |
| **E_lt200_lr001** | **8** | **200** | **0.001** | **60.54%** | **+2.01%** | **0.070** | **91.0%** | **+0.048** | **0.478** |
| E_lt500_lr001 | 8 | 500 | 0.001 | 61.30% | +2.77% | 0.074 | 90.1% | +0.047 | 0.475 |
| E_rank4_lt200_lr001 | 4 | 200 | 0.001 | 61.18% | +2.65% | 0.035 | 91.3% | +0.027 | 0.321 |

Baseline (ER, 10 ep): 58.53%

## The Breakthrough: Small teleport_lr Eliminates delta_norm Problem

With teleport_lr=0.001 (vs 0.01):
- delta_norm drops from 12–16 → **0.32–0.48** (30x reduction)
- avg_lt drops from 1.4–2.0 → **0.035–0.074** (40x reduction)
- cos+ rate IMPROVES: 62.5% → **91.3%**
- Accuracy crosses baseline: -29% → **+2.65%**

The smaller learning rate inside the teleport optimizer forces tiny LoRA perturbations.
These tiny perturbations are sufficient to modestly improve cos_sim (mean +0.027) while
causing essentially no loss disruption (lt ≈ 0.035). The weight perturbation is small
enough that future batches are unaffected.

## rank=4 vs rank=8 with lr=0.001

Both achieve essentially the same accuracy (61.18% vs 61.30%). This confirms that:
- teleport_lr, NOT rank, is the critical parameter
- rank=4 is sufficient when the perturbation is constrained (small lr)
- The earlier hypothesis that "higher rank = more freedom to find good points" was wrong:
  with lr=0.001, both ranks stay close to the initial point, and the cos_improve is small
  but positive in either case

## freq=1 Puzzle

E_freq1_lr001 (freq=1, 7358 triggers) = 60.49%
E_rank4_lt200_lr001 (freq=10, ~755 triggers) = 61.18%

**10x more teleport events gives the same (or slightly worse) accuracy.**

Possible explanations:
1. Each teleport improves cos_sim by only +0.027 on average — these improvements don't accumulate
2. With ER, teleport memory batch ≠ ER replay batch → alignment doesn't transfer to actual training
3. The cumulative delta_norm from 7358 small perturbations may add up

Hypothesis: With SGD (no ER batch mismatch), freq=1 should outperform freq=10 if explanation 2 is correct.
→ Testing in H11.

## Verdict

teleport_lr=0.001 turns a broken method into a modestly positive one (+2.65% over ER baseline).
The mechanism works: cos_sim improvements are real and loss-invariant. But the improvement
is small and frequency-invariant — suggesting the batch mismatch or gradient reset issue limits
the cumulative effect.

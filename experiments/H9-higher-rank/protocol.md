# H9: Higher LoRA Rank for Online Teleportation

## Hypothesis

All H7 experiments used lora_rank=4. The E6 log shows that with strict loss constraint
(lt_weight=10, n_steps=20), most teleport events have **negative cos_improve** — the optimizer
cannot find a direction in rank-4 LoRA space that simultaneously preserves loss AND improves
gradient alignment. Increasing rank to 8 or 16 gives more degrees of freedom on the loss
level set, making it more likely that a point with better cos_sim exists and can be found.

**Prediction**: With lora_rank=8 or 16, the fraction of teleport events with positive cos_improve
will increase (H7 E6: ~30% positive → expected >50%). If cos_improve reliably turns positive,
Class-IL accuracy should improve toward or above baseline (58.53%).

## Controlled Comparison (vs E5 which is best H7 result)

All hyperparameters fixed to E5 values:
- dataset: seq-cifar10, model: er, buffer_size=500
- lr=0.1, n_epochs=10, seed=42, batch_size=32
- teleport_mode=online, teleport_check_freq=10
- teleport_conflict_threshold=0.0
- teleport_lt_weight=50.0   (enforce loss invariance)
- teleport_online_steps=10
- teleport_lr=0.01
- teleport_memory_per_task=256
- teleport_reg=0.1

**Only variable: teleport_lora_rank**

## Experimental Design

| ID      | lora_rank | Δ from E5       | Prediction         |
|---------|-----------|-----------------|-------------------|
| E5-ref  | 4         | baseline (E5)   | 56.90% (observed) |
| E_rank8 | 8         | 2x more dims    | cos_improve↑, acc >56.90% |
| E_rank16| 16        | 4x more dims    | cos_improve↑↑, acc >58.53% |

## Metrics

- Primary: Class-IL mean accuracy after all tasks
- Secondary:
  - Fraction of triggers with positive cos_improve
  - Mean cos_improve per trigger
  - avg_lt (should stay ≈ same or lower)
  - Throughput (it/s)

## Dataset / Model

seq-cifar10, ER, buffer_size=500, lr=0.1, seed=42, n_epochs=10

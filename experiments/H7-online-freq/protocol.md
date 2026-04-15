# H7: Online Teleport — check_freq & diagnostic sweep

## Hypothesis
Setting `teleport_check_freq=1` (check every training step) will give teleportation the
maximum opportunity to reduce gradient conflict, potentially improving Class-IL accuracy
over the baseline ER.  If trigger rate is too high (unstable or slow), relaxing the
`teleport_conflict_threshold` below 0.0 can reduce triggers without changing frequency.

## Prediction
- check_freq=1 + threshold=0.0 will trigger teleport on ~10–40% of steps (TBD from diagnostic)
- Per-step teleport monitoring should show: cos_sim improves toward 0 after teleport, lt stays small
- Class-IL accuracy >= baseline ER at same epoch count

## Experimental Design

### Baselines
| ID    | config                              | epochs | purpose           |
|-------|-------------------------------------|--------|-------------------|
| E0-50 | ER baseline                         | 50     | reference (63.39%)|
| E0-10 | ER baseline                         | 10     | fast reference    |

### Online Teleport
| ID     | check_freq | threshold | n_steps | epochs | notes          |
|--------|-----------|-----------|---------|--------|----------------|
| E1-d   | 1         | 0.0       | 5       | 1      | diagnostic     |
| E1     | 1         | 0.0       | 5       | 10     | full run       |
| E2     | 1         | -0.3      | 5       | 10     | lower threshold|
| E3     | 10        | 0.0       | 5       | 10     | freq comparison|

## Metrics
- Primary: Class-IL mean accuracy after all tasks
- Secondary: teleport_trigger_count, cos_sim improvement (initial→final per trigger), lt values

## Dataset / Model
- seq-cifar10, ER, buffer_size=500, lr=0.1, seed=42

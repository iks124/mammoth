# H8: Verify Gradient Conflict → Forgetting Hypothesis

## Research Question
Does gradient conflict (cos_sim(g_old, g_new) < 0) actually CAUSE catastrophic
forgetting, or is it merely correlated / uncorrelated with forgetting?

All teleportation work (H1–H7) assumes: reduce gradient conflict → reduce forgetting.
If this assumption is false, all teleport variants are solving the wrong problem.

## Hypothesis
H8: Per-step gradient conflict (cos_sim) measured between old and new task gradients
is positively correlated with per-task forgetting on seq-cifar10.
- Low (negative) cos_sim during training on task T → higher forgetting of old tasks
- High (positive) cos_sim during training on task T → lower forgetting of old tasks

## Prediction
- Average cos_sim during task T training shows Pearson r < -0.3 with final accuracy
  on tasks 1..T-1 (lower cos_sim → lower accuracy = higher forgetting)
- Experiments with higher trigger rate (more gradient conflict) should show more
  forgetting per-task

## Experimental Design

### E0: Baseline ER (no teleport)
- seq-cifar10, ER, buffer_size=500, lr=0.1, seed=42, n_epochs=10
- Log cos_sim at every step (detection-only, threshold=1000 so never triggers)
- Collect per-task accuracy trajectory

### E1: Multi-seed correlation study
- Same as E0 but seeds 42, 123, 456
- Compute: per-task mean cos_sim vs per-task forgetting
- Expected output: scatter plot + Pearson correlation

### E2: Forced low-conflict baseline (for contrast)
- Reduce learning rate to 0.01 (smaller gradient updates → potentially different conflict pattern)
- Compare cos_sim distribution and forgetting with E0

## Implementation Plan

1. Add `--teleport_detect_only` flag: runs detection but never merges LoRA
2. Log per-step cos_sim with step number and current task ID
3. After training: compute per-task forgetting and correlate with mean cos_sim per task
4. Plot: (a) cos_sim over training steps colored by task, (b) scatter per-task cos_sim vs forgetting

## Metrics
- Primary: Pearson r(cos_sim, forgetting) per task
- Secondary: cos_sim distribution per task, forgetting per task

## Dataset / Model
- seq-cifar10, ER, buffer_size=500, lr=0.1, seed=42

## Success Criteria
- r < -0.3: hypothesis supported → teleport approach is on the right track
- |r| < 0.1: hypothesis NOT supported → fundamental pivot needed
- r > 0.3: unexpected (higher conflict → less forgetting?) → need new mental model

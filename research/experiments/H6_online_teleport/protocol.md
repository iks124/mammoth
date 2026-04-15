# H6 Protocol: Online Gradient Conflict Detection + Teleportation

**Status:** Running  
**Date:** 2026-04-15  
**Hypothesis:** Online teleportation (mid-training) reduces forgetting more reliably than task-boundary teleportation

## Prediction

均值 Class-IL > 45.64% (baseline), std < 3.0%

## Rationale

- 之前所有方案（H1/H1b/H5）都在 task boundary 执行单次 teleportation，违反漏洞5：
  零阶保证无法覆盖整个训练轨迹
- COST/BOOST 都把 teleportation 整合进优化过程（训练中持续检测并纠正）
- H6 在每 k=50 步检测 cos_sim(g_new, g_old)；若 < 0 则触发 mini LoRA teleportation（5步）
- 关键：L_t 约束确保每次 teleportation delta 极小（不破坏当前训练状态）

## Experiment Config

```bash
python utils/main.py \
  --model er \
  --dataset seq-cifar10 \
  --lr 0.1 \
  --n_epochs 1 \
  --buffer_size 500 \
  --seed {42,1,2,3} \
  --teleport 1 \
  --teleport_mode online \
  --teleport_check_freq 50 \
  --teleport_online_steps 5 \
  --teleport_conflict_threshold 0.0 \
  --teleport_lora_rank 2 \
  --teleport_lr 1e-3 \
  --teleport_reg 0.01 \
  --teleport_lt_weight 1.0 \
  --teleport_approx_layers 2 \
  --teleport_memory_per_task 256 \
  --nowand --non_verbose
```

## Baseline (no teleport)

| seed | class-IL |
|------|----------|
| 42   | 43.71%   |
| 1    | 45.83%   |
| 2    | 45.96%   |
| 3    | 47.04%   |
| mean | 45.64%   |
| std  | 1.34%    |

## Results

| seed | baseline | H6 (online) | delta |
|------|----------|-------------|-------|
| 42   | 43.71%   | 53.96%      | +10.25% |
| 1    | 45.83%   | 48.51%      | +2.68% |
| 2    | 45.96%   | 41.57%      | -4.39% |
| 3    | 47.04%   | 44.15%      | -2.89% |
| **mean** | **45.64%** | **47.05%** | **+1.41%** |
| **std** | **1.34%** | **5.43%** | — |

**结论：** 均值 +1.41%，但方差从 1.34% 爆炸到 5.43%。与 H5 模式相同：部分 seed 大幅提升，部分 seed 下降。
- Teleportation 被频繁触发（每个 epoch 约 6 次），delta_norm ≈ 0.3-0.5，累积效应大
- 部分成功（seed=42, +10.25%），部分失败（seed=2, -4.39%）
- 与 baseline std 相比，方差增大 4x

**初步分析：**
- 在某些初始化/任务序列下，online teleportation 能有效减少冲突（seed=42）
- 在其他情况下，频繁的小步 LoRA 扰动破坏了训练动态（seed=2,3）
- cos_sim 只部分被解决：很多事件从负值只改善到 -0.1~0.0（未真正变正）

**下一步：**
- H6b：减少触发频率（threshold=-0.3，只在严重冲突时触发）
- H6c：增加 post-teleport guard（检查 old-task loss 没有恶化）
- H6d：减小 delta_norm（更强的 reg 或更少步骤）

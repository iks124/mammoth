# Findings: Teleportation + Continual Learning

**项目开始时间：** 2026-04-14  
**核心问题：** 能否用 Symmetry Teleportation 减少 CL 中的灾难性遗忘？

---

## Current Understanding（当前认知，2026-04-14 outer loop）

### 已否定的方向

1. **ReLU scaling + grad_norm² 最小化**：精度下降 -2.84%，优化不收敛

2. **ReLU scaling + cos_sim（mini-batch）**：梯度噪声冻住参数，无效

3. **ReLU scaling + cos_sim（全量 batch）**：单 seed +2.2%，多 seed 均值 -1.51%，不稳定

4. **LoRA + COST sharpness 目标**（H1）：L_t 在 B=0 时梯度=0，L_g 爆炸，LoRA norm→15000，网络崩溃 ~10%

5. **LoRA + cos_sim**（H1b）：cos_sim 优化成功（-0.13→0.99），但 delta_norm=3，旧任务精度 -12~27%，均值 41.93%（**-3.70% vs baseline**）

### 核心矛盾（已证伪的假设）

> **"找到一个 cos_sim 更高的参数点就能减少遗忘"**

这个假设在两种参数化下都失败了：
- **ReLU scaling**：输出不变（无损），但512维自由度不足以实质性改变 cos_sim，轻微扰动效果随机
- **LoRA**：有足够自由度将 cos_sim 从-0.13推到0.99，但 delta_norm=3 破坏了已学表征，比不 teleport 更差

**根本矛盾**：改变 cos_sim 需要大幅改变参数 → 大幅改变参数破坏旧任务表征 → 净效果为负

这与 MTL（COST 成功的场景）的本质区别：
- MTL：teleportation 后继续训练所有任务，自然恢复
- CL：旧任务被冻结，teleportation 造成的损坏不可恢复（ER 只有少量 buffer，无法完全恢复）

---

## Patterns and Insights

1. **输出不变性是 CL teleportation 的必要条件**，不是可选的。只有 ReLU scaling 提供严格输出不变性，但其自由度太小。两者无法同时满足。

2. **cos_sim 在所有实验中都不是有效的 forgetting 代理**：改善 cos_sim 从未稳定地改善 Class-IL。

3. **COST 的成功依赖于 MTL 的 online recovery 机制**，这在 CL 中不存在。

4. **漏洞1（cos_sim≠forgetting）和漏洞2（自由度）之间存在 trade-off**：增大自由度使 cos_sim 可优化，但同时放大了漏洞1的危害。

---

## Lessons and Constraints

- `--nowand` 不是 Mammoth CLI 参数
- ER + n_epochs=1 单 seed 方差极大，需要多 seed（3~4个）
- teleportation 时序：必须在任务 t 训练前执行
- LoRA+cos_sim：reg=0.01 → delta_norm=3.0 → 旧任务精度 -12%
- COST sharpness：L_t gradient=0 at B=0 → 爆炸，无法使用

---

## Open Questions（开放问题）

1. **是否存在一种变换**，既有足够大的自由度，又能保证输出不变？（除了 ReLU 缩放之外）
2. **teleportation 的时机**：训练中间（每 epoch 后）是否比训练前更有效？
3. **完全不同的方向**：不再用 teleportation，改用 LoRA 做其他 CL 增强？

---

## Hypotheses（假设列表）

### ❌ H1：COST sharpness — 失败（LoRA 爆炸）
### ❌ H1b：LoRA+cos_sim — 失败（delta_norm=3，损坏旧任务，-3.70%）

### H4（新）：输出不变 LoRA（constrained）
- **假设**：给 LoRA 加显式输出不变约束（KL/MSE on old memory outputs），使 delta_norm 被压制到 <0.1，cos_sim 在此约束下小幅改善，净效果为正
- **预测**：与 baseline 持平或小幅提升（+0~1%）
- **方法**：augmented Lagrangian 或 penalty method：`-cos_sim + λ_kl * KL(p(θ+ΔΘ)||p(θ)) + λ_reg * ||BA||²`
- **状态**：待实现

### H5（新）：放弃 cos_sim，改用 intra-task LoRA fine-tuning
- **假设**：在每个任务边界，用 LoRA 在旧任务 memory 上做少量 fine-tuning（最小化旧任务 loss），相当于精确的 gradient 步骤但只改动 LoRA 子空间，然后 merge，减少遗忘
- **预测**：均值 > 46%
- **方法**：`L = L_old(θ+ΔΘ)` + `λ * ||BA||²`，在 task t 训练结束后执行（修复旧任务 loss）
- **状态**：待实现（这和 GEM/A-GEM 思想有共通之处，但用 LoRA 投影）

---

## Experiment Trajectory

| Exp ID | 方法 | 均值 Class-IL（4 seed） | delta |
|--------|------|------------------------|-------|
| baseline | no teleport | 45.64% | — |
| ReLU+cos seed42 only | ReLU scaling | 46.57% | +0.93% |
| ReLU+cos multi-seed | ReLU scaling | 44.13% | -1.51% |
| H1 COST sharpness | LoRA+sharpness | ~10% | **崩溃** |
| H1b LoRA+cos_sim | LoRA+cos_sim | 41.93% | **-3.70%** |
| H4 (planned) | constrained LoRA | — | — |
| H5 (planned) | LoRA finetune old | — | — |

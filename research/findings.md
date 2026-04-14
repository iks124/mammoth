# Findings: Teleportation + Continual Learning

**项目开始时间：** 2026-04-14  
**核心问题：** 能否用 Symmetry Teleportation 减少 CL 中的灾难性遗忘？

---

## Current Understanding（当前认知，2026-04-14 outer loop v2）

### 已否定的方向（按时间序）

1. **ReLU scaling + grad_norm² 最小化**：精度下降 -2.84%，优化不收敛

2. **ReLU scaling + cos_sim（mini-batch）**：梯度噪声冻住参数，无效

3. **ReLU scaling + cos_sim（全量 batch）**：单 seed +2.2%，多 seed 均值 -1.51%，不稳定

4. **LoRA + COST sharpness 目标**（H1）：L_t 在 B=0 时梯度=0，L_g 爆炸，LoRA norm→15000，网络崩溃 ~10%

5. **LoRA + cos_sim**（H1b）：cos_sim 优化成功（-0.13→0.99），但 delta_norm=3，旧任务精度 -12~27%，均值 41.93%（**-3.70% vs baseline**）

6. **LoRA repair on old memory**（H5）：均值 44.75%（**-0.89%**），std=4.49%（baseline std=1.34%）。seed=42/1 提升 +3~5%，但 seed=2/3 大幅下降，尤其 seed=3: -9.51%。**方差爆炸**。

### 核心矛盾（已证伪的假设）

> **假设1（已证伪）："找到一个 cos_sim 更高的参数点就能减少遗忘"**

> **假设2（已证伪）："在旧任务 memory 上做 LoRA fine-tuning 能修复遗忘"**

两个假设都在 CL 中失败，根本原因相同：**在 CL 中修改参数无法自恢复**。

- **MTL（COST 成功）**：teleportation 后继续训练所有任务，对任何微小破坏都能在线恢复
- **CL（失败场景）**：旧任务数据不再可访问（除了 ER 的少量 buffer），任何参数修改的损坏都是永久的

**H5 高方差的解释**：LoRA repair 本质上是在 LoRA 子空间做梯度步骤，优化旧任务 loss。当旧任务表征恰好与新任务发生冲突时（seed=3），repair 破坏了新任务刚建立的表征，导致大幅下降。这是随机种子决定的，不可控。

### 系统性结论（outer loop v2）

经过 6 种方法测试，有一个统一的失败模式：

**"任何参数修改在 CL 中都会增加不稳定性，因为没有在线恢复机制"**

具体表现：
- 修改幅度小（ReLU scaling）→ 效果不显著，方差随机
- 修改幅度大（LoRA+cos_sim）→ 破坏旧任务表征，均值下降
- 修改目标明确（H5 LoRA repair）→ 均值轻微下降，但**方差爆炸**（4.49% vs 1.34%）

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
- LoRA repair（H5）：即使 conservative（steps=5, lr=1e-4, reg=1.0），方差从 1.34% 爆炸到 4.49%；某些 seed 下 -9% 下降不可接受
- **方差增加是 CL teleportation 失败的新信号**：不仅看均值，std 也是关键指标

---

## Open Questions（开放问题）

1. **H4 是否可行**？输出不变约束（MSE/KL on old memory）能否让 LoRA 在不破坏旧表征的前提下改变梯度几何？
   - 理论问题：若输出严格不变，梯度几何能有多大改变？可能答案是"改变不够大，net effect≈0"
   - 若此假设也失败，则与 ReLU scaling 的失败原因相同（自由度不足）

2. **"输出不变" vs "损失不变"**：这两个约束之间有多大差别？H4 是否比 ReLU scaling 有本质上更大的自由度？

3. **是否应该结束 teleportation 方向，改写 negative result 论文？**
   - 现有 6 种方法失败，有清晰的理论解释，足以发表 negative result
   - 或者找到一个有效的方向再继续

---

## Hypotheses（假设列表）

### ❌ H1：COST sharpness — 失败（LoRA 爆炸）
### ❌ H1b：LoRA+cos_sim — 失败（delta_norm=3，损坏旧任务，-3.70%）

### H4（新）：输出不变 LoRA（constrained）
- **假设**：给 LoRA 加显式输出不变约束（KL/MSE on old memory outputs），使 delta_norm 被压制到 <0.1，cos_sim 在此约束下小幅改善，净效果为正
- **预测**：与 baseline 持平或小幅提升（+0~1%）
- **方法**：augmented Lagrangian 或 penalty method：`-cos_sim + λ_kl * KL(p(θ+ΔΘ)||p(θ)) + λ_reg * ||BA||²`
- **状态**：待实现

### ❌ H5：LoRA repair on old memory — 失败（高方差，均值 44.75%，-0.89%）
- **结果**：seed=42: 48.71%, seed=1: 48.32%, seed=2: 44.45%, seed=3: 37.53%
- **均值**: 44.75%（vs baseline 45.64%，**-0.89%**），std=**4.49%**（baseline std=1.34%）
- **分析**：两个 seed 明显超出 baseline（+3~5%），另两个 seed 大幅下降（尤其 seed=3: -8.11%）
- **关键问题**：方差爆炸。LoRA repair 在某些初始化/任务序列下有效，其他情况下破坏了当前任务学习
- **根本原因**：修复旧任务 loss 的 LoRA 步骤可能与当前任务表征发生干扰，当旧任务与新任务特征空间冲突较大时失败

---

## Experiment Trajectory

| Exp ID | 方法 | 均值 Class-IL（4 seed） | delta |
|--------|------|------------------------|-------|
| baseline | no teleport | 45.64% | — |
| ReLU+cos seed42 only | ReLU scaling | 46.57% | +0.93% |
| ReLU+cos multi-seed | ReLU scaling | 44.13% | -1.51% |
| H1 COST sharpness | LoRA+sharpness | ~10% | **崩溃** |
| H1b LoRA+cos_sim | LoRA+cos_sim | 41.93% | **-3.70%** |
| H5 conservative | LoRA repair old (reg=1.0,steps=5) | 44.75%±4.49% | **-0.89%，高方差** |
| H4 (planned) | constrained LoRA+output invariance | — | — |

# Findings: Teleportation + Continual Learning

**项目开始时间：** 2026-04-14  
**核心问题：** 能否用 Symmetry Teleportation 减少 CL 中的灾难性遗忘？

---

## Current Understanding（当前认知）

### 已否定的方向

1. **ReLU scaling + grad_norm² 最小化**（阶段1）：优化不收敛，精度下降 -2.84%

2. **ReLU scaling + cos_sim 最大化（mini-batch）**：mini-batch 噪声使 log_t 被冻住（max|log_t| ≤ 0.022），无效

3. **ReLU scaling + cos_sim 最大化（全量 batch）**：reg=0.1 时单 seed 有 +2.2%，但多 seed 验证均值 -1.51%，不稳定

4. **ReLU scaling 的根本瓶颈**：
   - 自由度 512 维 / 网络 11M 参数 = 0.005%
   - cos_sim ≠ forgetting reduction（漏洞1）
   - 单点优化无轨迹保证（漏洞5）

### 关键文献洞见：COST（2503.04046）

COST 在 MTL 场景中成功应用 teleportation，关键差异：

1. **LoRA 替代 ReLU scaling**：自由度从 512 提升到 rank×model_dim（数量级差异）
2. **双目标**：loss invariance + gradient maximization（sharpness），而非 cos_sim
3. **HTR 策略**：解决 teleportation 后 Adam momentum 不匹配问题
4. **条件触发**：检测冲突才 teleport，不是每步都做

---

## Patterns and Insights（规律与洞见）

- ReLU scaling 的核心问题是**自由度不足**，而非目标函数的选择
- cos_sim 作为代理目标的根本问题：幅度信息丢失 + 零阶保证
- **LoRA** 是可行的高自由度替代方案（COST 验证有效）
- CL 和 MTL 的任务冲突结构高度相似：old tasks = A, new task = B

---

## Lessons and Constraints（教训）

- `--nowand` 不是 Mammoth 的 CLI 参数，会报错
- ER + seq-cifar10 + n_epochs=1 的单 seed 方差极大（43%~49%），需要多 seed
- teleportation 时序：必须在任务 t **训练前**执行（已修复）
- 全量 batch 对 teleportation 稳定性至关重要（mini-batch 下梯度方向不一致）

---

## Open Questions（开放问题）

1. **COST-CL 假设是否成立**：LoRA teleportation (loss invariant on old memory + gradient max on new task) 能否减少 CL forgetting？
2. **HTR 在 CL 中的作用**：任务切换时调制 Adam momentum 是否有额外收益？
3. **触发条件**：CL 中是否需要冲突检测触发，还是每个任务都做？
4. **超参数**：γ、LoRA rank、teleportation 步数的敏感性？

---

## Hypotheses（假设列表）

### H1（主要）：COST-CL —— LoRA teleportation for CL
- **假设**：在任务 t 训练前，用 LoRA (rank=4) 对 shared backbone teleport，目标 = L_old_invariance - γ · L_new_sharpness，能改善 ER 在 seq-cifar10 上的 Class-IL 精度
- **预测**：均值 > 45.64%（当前 no-teleport baseline），多 seed 稳定
- **状态**：待实验

### H2：HTR 的贡献
- **假设**：在 H1 基础上加 HTR 策略（调制任务 t 开始时的 Adam momentum），有额外收益
- **预测**：在 H1 基础上 +1~2%
- **状态**：待 H1 验证后

### H3：触发条件
- **假设**：用冲突检测（而非每个任务都 teleport）触发，可以降低开销同时保持效果
- **状态**：待 H1 验证后

---

## Experiment Trajectory

| Exp ID | 假设 | 配置 | Class-IL（均值4seed） | delta vs baseline |
|--------|------|------|-----------------------|-------------------|
| baseline | - | ER, no teleport, seeds 42/1/2/3 | 45.64% | - |
| old-tp-cos | cos_sim | ReLU scaling, reg=0.1, seed 42 | 46.57% | +2.86% |
| old-tp-multi | cos_sim | ReLU scaling, multi-seed | 44.13% | -1.51% |
| H1-v1 | COST-CL | **待运行** | - | - |

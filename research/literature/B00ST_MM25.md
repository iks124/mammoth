# B00ST: Conflict-Buffering Optimization by Symmetry Teleportation for Deep Long-Tailed Recognition

**Venue:** ACM Multimedia 2025 (MM '25), Dublin, Ireland  
**Authors:** Mianzimei Yang, Zhipeng Zhou, Jin Zhang, Yuanhao Pu, Hong Xie, Defu Lian  
**Affiliation:** USTC + NTU Joint WeBank Research Institute  
**DOI:** https://doi.org/10.1145/3746627.3755153  
**Tags:** deep long-tailed recognition (DLTR), symmetry teleportation, gradient conflict, LoRA

---

## 核心问题

**Deep Long-Tailed Recognition（DLTR）中的梯度冲突**：在类别分布极度不均匀的数据集中（如 CIFAR-10-LT imbalance ratio=200），头部类别样本数量远超尾部类别，训练时头部类别主导梯度方向，尾部类别梯度被压制甚至反转，导致尾部类别精度极差。

---

## 为什么不用 MOO

现有 MOO-based DLTR 方法（GBG, MOOSF, PLOT）存在两个问题：
1. 性能高度依赖强数据增强（关闭 auto-augmentation 后 GBG/MOOSF 大幅下降）
2. 内存消耗极大（PLOT 消耗 ERM 的 2.7~3.5 倍显存）

B00ST 改用 symmetry teleportation 替代 MOO，计算更轻量。

---

## Teleportation 在 B00ST 中的使用方式

### 1. 触发条件：冲突检测

将所有类别划分为 M 组，计算各组梯度 G = {g_i}。当任意两组梯度 cos_sim < 0 时触发：
```
cos_sim(g_i, g_j) < 0，g_i, g_j ∈ G   （Eq. 1）
```

### 2. Teleportation 机制（LoRA）

`θ' = θ + Δθ = θ + BA`（B: d×r，A: r×d，rank r 极小）

两个目标：

**L_t（Loss Invariance）**：最小化 teleportation 前后各组 loss 变化
```
L_t = Σ_{i=1}^{M} |L_orig^(i) - L_cur^(i)|   （Eq. 3）
```
L_orig^(i) 是 teleportation 开始前各组 loss（冻结，不更新）。

**L_g（Balanced Gradient Maximization）**：最大化各组梯度范数的加权和
```
L_g = Σ_{i=1}^{M} R_i · ||E[∇h^(i)(θ+rv)]q/r||²   （Eq. 6）
```
其中 R_i 是重加权因子（Eq. 5），给梯度小的组（尾部类别）更大权重：
```
R = M · softmax( (Σ ||g_j|| / ||g_i||)_{i=1}^{M} )
```
这使尾部类别梯度在 teleportation 后得到更大提升。

**总目标**：`L = L_t - α * L_g`（Eq. 7）

### 3. Trajectory Reuse Strategy（TRS）

与 COST 的 HTR 几乎完全相同：teleportation 后动量状态与新参数点不匹配，B00ST 比较旧梯度方向与 teleportation 方向的 cos_sim，若对齐则复用历史动量，否则丢弃：
```
若 cos_sim(θ_old → θ_new, ∇h_l) > 0：复用旧动量 m_{i+1} = γm_i + θ_i' 
```

### 4. 执行流程

```
训练过程中：
if 检测到冲突:
    冻结 θ，初始化 LoRA (A, B)
    Step 1: 计算重加权 R（平衡各组梯度范数）
    Step 2: 用 L = L_t - α * L_g 训练 ΔΘ = BA
    Merge: θ_new = θ + BA
    if TRS 条件满足: 复用历史动量
继续正常训练（在新参数点上）
```

---

## B00ST vs COST 对比

| 维度 | COST（MTL） | B00ST（DLTR） |
|---|---|---|
| 任务结构 | K 个任务同时训练 | 所有类别（头/尾）同时训练 |
| 冲突类型 | 任务间梯度冲突 | 类别组间梯度冲突（头>尾） |
| L_t | 所有任务 loss 不变 | 所有类别组 loss 不变 |
| L_g | 随机扰动估计 sharpness | 加权梯度范数（尾部类别加权更重） |
| 触发条件 | 超过 K/2 任务与均值梯度冲突 | 任意两组 cos_sim < 0 |
| TRS/HTR | 完全相同机制 | 完全相同机制 |
| 训练后 | 继续训练所有任务 | 继续训练所有类别 |
| **数据可访问性** | **所有任务数据始终可见** | **所有类别数据始终可见** |

**本质相同点**：两者都是在**全数据可访问的并发训练**中解决梯度几何问题。

---

## 实验结果

### CIFAR-10-LT / CIFAR-100-LT（Table 2）

| 方法 | CIFAR10-LT (200/100/50) | CIFAR100-LT (200/100/50) |
|---|---|---|
| GBG | -/80.08/- | 36.84/41.59/47.48 |
| MOOSF | 44.82/82.06/85.16 | - |
| PLOT | 80.08/83.35/85.90 | 45.61/49.50/53.05 |
| **B00ST** | **80.06/84.19/87.21** | **44.96/50.21/54.87** |

### ImageNet-LT（Table 1）
- B00ST: 56.8% Overall（ResNeXt-50），超过 GLMC(56.3%)、MiSLAS(52.1%)

### Ablation（Table 3，CIFAR100-LT，ratio=100）
| L_t | L_g | TRS | Acc |
|---|---|---|---|
| ✗ | ✗ | ✗ | 48.46 |
| ✓ | ✗ | ✗ | 49.82 |
| ✓ | ✓ | ✗ | 50.02 |
| ✓ | ✓ | ✓ | 50.21 |

每个组件都有正贡献，L_t（loss invariance）贡献最大。

### Conflict Resolution 验证（Figure 8）
Teleportation 后，冲突（min cos_sim < 0）基本消失，即使少数 case 冲突仍存在，也显著减轻。

---

## 对 CL 的参考意义

B00ST 成功的必要条件（与 COST 相同）：
1. **所有数据始终可访问**：teleportation 后的正常训练步骤会自然修复任何微小损坏
2. **冲突是并发的**：所有任务/类别同时参与梯度计算，冲突来自同一个 batch 内
3. **LoRA merge 后继续训练**：merge 引入的扰动通过后续梯度步骤得到在线校正

CL 缺少的：
- 旧任务数据不再完整可访问（只有 ER buffer ~500 samples）
- merge 引入的损坏无法通过"继续训练所有任务"来恢复
- 冲突的性质是"序列化"的，不是"并发"的

---

## 已知局限（论文明确提到）

- 依赖数据分组（M 组），M 的选择影响性能
- α 超参数需要调整
- 当 imbalance ratio 较小时（ratio=200 on CIFAR10-LT），PLOT 略优于 B00ST

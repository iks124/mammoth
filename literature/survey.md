# Literature Survey: Gradient-Based Teleportation for Continual Learning

## Direct Inspirations

### COST: Continual Optimization with Symmetry Teleportation for MTL
- **Ref**: Zhou et al., arXiv:2503.04046, March 2025
- **Setting**: Multi-Task Learning (all tasks available simultaneously)
- **Method**: Detects gradient conflicts in MTL; uses LoRA to find loss-equivalent
  parameter points with reduced conflict; merges LoRA back post-teleport
- **Results**: Effective for MTL; plug-and-play with other MTL methods
- **Limitations**: Additional training cost; less competitive on regression tasks;
  LoHa/OFT do NOT improve on baseline (complex PEFT can hurt generalizability)
- **Key difference from our work**: MTL has all tasks available simultaneously;
  CL has sequential access only. The conflict structure is fundamentally different.

### Symmetry Teleportation for Accelerated Optimization
- **Ref**: Zhao et al., arXiv:2205.10637, NeurIPS 2022
- **Setting**: Single-task optimization
- **Method**: ReLU scaling symmetry — multiply pre-activation by t, post-activation by 1/t
  to find equivalent points with better curvature (sharper minima → faster convergence)
- **Relevance**: Original teleportation idea; we adapted to CL gradient alignment

---

## Key Related Work on Gradient Similarity and Forgetting

### Learning the Mechanism of Catastrophic Forgetting: A Perspective from Gradient Similarity
- **Ref**: arXiv:2601.21577, January 2026
- **Setting**: LLM continual fine-tuning
- **Key Finding**: Proves strongly negative gradient similarity is a **fundamental cause** of forgetting
  - 50–75% of neurons are "conflicting neurons" (negative gradient similarity)
  - 25–50% are "collaborative neurons" (positive gradient similarity)
- **Method**: CNL — freeze conflicting neurons, train only collaborative ones
- **Results**: Zero forgetting in-set; 59.1–81.7% reduction out-of-set (across 5 LLMs, 4 datasets)
- **Relation to our work**:
  - CONFIRMS our H8 finding: ubiquitous gradient conflict (78.9% of steps = ~conflicting neuron fraction)
  - BUT their solution is to FREEZE conflicting neurons — not align gradients
  - Our approach tried to RE-ALIGN gradients via teleportation → fundamentally different strategy
  - Key insight: freezing is more robust than re-alignment because re-alignment requires
    changing the parameter, which has side effects on all other batches (our Flaw 5)

### Gradient Episodic Memory (GEM)
- **Ref**: Lopez-Paz & Ranzato, NeurIPS 2017
- **Method**: Project gradient updates to not increase loss on replay buffer
  (QP constraint: g'g_old ≥ 0)
- **Relevance**: Closest prior work to our idea — also uses gradient alignment constraint
- **Key difference**: GEM constrains the SGD gradient direction directly (first-order);
  we tried to change the PARAMETER via LoRA to improve gradient alignment (second-order)

### A-GEM (Averaged GEM)
- **Ref**: Chaudhry et al., ICLR 2019
- **Method**: Simplified GEM using average gradient over episodic memory
- **Results**: More efficient than GEM; still competitive
- **Relevance**: Shows gradient projection can work if applied to the right quantity

### PCGrad: Gradient Surgery for Multi-Task Learning
- **Ref**: Yu et al., NeurIPS 2020
- **Method**: Project conflicting task gradients onto the normal plane of each other
- **Setting**: MTL (simultaneous tasks)
- **Key insight**: Gradient conflict correction helps MTL; but not directly tested in sequential CL

---

## LoRA for Continual Learning (2025 context)

### TreeLoRA (ICML 2025)
- **Ref**: lamda.nju.edu.cn/qianyy/paper/ICML25_TreeLoRA.pdf
- **Method**: Layer-wise LoRA adapters organized in hierarchical tree based on gradient similarity
- **Relevance**: Also uses gradient similarity for CL; different approach (task-specific LoRAs)

### KeepLoRA (arXiv:2601.19659)
- **Method**: Initialize LoRA updates from first training step gradients; project into residual subspace
- **Relevance**: Shows gradient-aware LoRA initialization can help CL

### PLAN (ICCV 2025)
- **Method**: Proactive low-rank allocation for CL
- **Relevance**: Another LoRA-based approach to CL

---

## Key Differentiators of Our Work

1. **First systematic test of online gradient alignment teleportation for sequential CL**
   - COST tested MTL only (simultaneous task access)
   - We test the harder case: sequential, no future task access

2. **Identified Flaw 5 (Zero-order Trap)**
   - Loss invariance at one batch ≠ trajectory invariance
   - This is a structural limitation not discussed in prior teleportation work

3. **H8: Quantified ubiquity of gradient conflict in sequential CL**
   - 78.9% of training steps show cos_sim(g_old, g_new) < 0
   - Consistent with 2601.21577's 50-75% conflicting neuron finding
   - But shows that single-step cos_sim is too noisy a signal for teleportation

4. **Coherent negative result with mechanistic explanation**
   - Not just "it doesn't work"; shows WHY at multiple levels

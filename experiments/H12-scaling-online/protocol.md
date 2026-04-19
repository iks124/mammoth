# H12: Scaling-Online Teleportation — lt-cos Tradeoff Test

## Hypothesis

The lt-cos tradeoff is the bottleneck limiting LoRA online teleportation effectiveness.
LoRA with lt_weight=200 can only achieve cos_improve=+0.007 because the loss-invariance
penalty severely constrains the optimization range.

ReLU scaling symmetry provides EXACT loss invariance (mathematical guarantee, no penalty
needed), allowing unlimited cos_sim optimization. If the lt-cos tradeoff is truly the
bottleneck, scaling_online should achieve cos_improve >> +0.007 and show forgetting reduction.

## Prediction

- SGD_sc_f1_lr01: cos_improve > +0.05 (at minimum); Class-IL improvement if hypothesis holds
- SGD_sc_f1_lr1: even larger cos_improve; may be unstable

## Falsification

- If cos_improve is still < +0.02 → scaling DOF (1920 log_t params) insufficient; different problem
- If cos_improve > +0.05 but Class-IL unchanged → gradient signal quality is the real bottleneck

## Setup

Fixed (same as H11 SGD):
  dataset=seq-cifar10, lr=0.1, n_epochs=10, seed=42
  teleport_memory_per_task=256, threshold=0.0, freq=1
  model=sgd (no ER)

| ID | teleport_mode | lr_t | reg | steps |
|----|---------------|------|-----|-------|
| SGD_sc_f1_lr01 | scaling_online | 0.01 | 0.1 | 10 |
| SGD_sc_f1_lr1  | scaling_online | 0.1  | 0.1 | 10 |

Reference baselines (H11):
  SGD_base: 19.45%
  SGD_f1 (LoRA lr001): 19.44%, cos_improve=+0.007, lt=0.015

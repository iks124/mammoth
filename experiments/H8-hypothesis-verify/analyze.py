"""
H8 Analysis: Does gradient conflict correlate with catastrophic forgetting?

Parses training logs from detection-only runs. For each task, computes
mean cos_sim during training and correlates with per-task forgetting.

Usage:
    python experiments/H8-hypothesis-verify/analyze.py \
        --log experiments/H8-hypothesis-verify/logs/E0_seed42_detect.log \
        --out experiments/H8-hypothesis-verify/results/E0_seed42.csv

Output:
    - CSV: task, n_checks, mean_cos_sim, neg_frac, peak_acc, final_acc, forgetting
    - Pearson r(cos_sim, forgetting) — the main metric
"""

import re
import sys
import argparse
import csv
import os
from collections import defaultdict


def parse_log(log_path):
    """
    Parse log file and extract:
    - cos_sim measurements per task: {task_id: [cos_sim, ...]}
    - per-task accuracy snapshots after each task's training
    """
    cos_by_task = defaultdict(list)
    # Each evaluation produces: Raw accuracy values: Class-IL [acc0, acc1, ...]
    # The n-th evaluation (after task n-1) lists accs for tasks 0..n-1
    accuracy_snapshots = []  # list of lists: snapshot[i] = [acc_t0, acc_t1, ..., acc_ti]

    with open(log_path) as f:
        for line in f:
            # Detect cos_sim log: [Teleport-Detect] task=N step=S cos_sim=C threshold=T
            m = re.search(r'\[Teleport-Detect\] task=(\d+) step=(\d+) cos_sim=([+-]?\d+\.\d+)', line)
            if m:
                task_id = int(m.group(1))
                cos_sim = float(m.group(3))
                cos_by_task[task_id].append(cos_sim)
                continue

            # Accuracy snapshot: "Raw accuracy values: Class-IL [a, b, c, ...]"
            m = re.search(r'Raw accuracy values: Class-IL \[([^\]]+)\]', line)
            if m:
                vals_str = m.group(1)
                accs = [float(x.strip()) for x in vals_str.split(',')]
                accuracy_snapshots.append(accs)
                continue

    return cos_by_task, accuracy_snapshots


def compute_forgetting(accuracy_snapshots):
    """
    Compute per-task forgetting.

    For task t: peak accuracy = first time task t appears in a snapshot
                final accuracy = accuracy in the LAST snapshot
    forgetting[t] = peak_acc[t] - final_acc[t]
    """
    if not accuracy_snapshots:
        return {}, {}, {}

    n_tasks = len(accuracy_snapshots[-1])
    peak_acc = {}
    for snap in accuracy_snapshots:
        for t, acc in enumerate(snap):
            if t not in peak_acc:
                peak_acc[t] = acc

    final_accs = accuracy_snapshots[-1]
    final_acc = {t: final_accs[t] for t in range(len(final_accs))}

    forgetting = {}
    for t in peak_acc:
        if t in final_acc:
            forgetting[t] = peak_acc[t] - final_acc[t]

    return forgetting, peak_acc, final_acc


def pearson_r(xs, ys):
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    den_y = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if den_x < 1e-10 or den_y < 1e-10:
        return float('nan')
    return num / (den_x * den_y)


def analyze(log_path, out_path=None, verbose=True):
    cos_by_task, accuracy_snapshots = parse_log(log_path)
    forgetting, peak_acc, final_acc = compute_forgetting(accuracy_snapshots)

    if verbose:
        print(f"\n=== H8 Analysis: {os.path.basename(log_path)} ===\n")
        print(f"Tasks with cos_sim data: {sorted(cos_by_task.keys())}")
        print(f"Evaluation snapshots: {len(accuracy_snapshots)}")
        if accuracy_snapshots:
            print(f"Final snapshot: {accuracy_snapshots[-1]}")

        print("\n--- cos_sim per task ---")
        for t in sorted(cos_by_task.keys()):
            vals = cos_by_task[t]
            mean_c = sum(vals) / len(vals)
            neg_frac = sum(1 for v in vals if v < 0) / len(vals)
            print(f"  task={t}: n={len(vals):4d}  mean={mean_c:+.4f}  "
                  f"neg_frac={neg_frac:.1%}  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}")

        print("\n--- Per-task accuracy & forgetting ---")
        for t in sorted(peak_acc.keys()):
            fa = final_acc.get(t, float('nan'))
            f = forgetting.get(t, float('nan'))
            print(f"  task={t}: peak={peak_acc[t]:.2f}%  final={fa:.2f}%  "
                  f"forgetting={f:+.2f}%")

    # Correlation: tasks that have cos_sim data (during training) AND forgetting measure
    # Task 0 has no cos_sim data (detection only starts from task 1)
    # Tasks 1..n-1 have cos_sim data; forgetting is measured for tasks 0..n-2
    # Useful overlap: tasks 1..n-2 (trained while old tasks exist, and ARE old tasks at end)
    tasks_with_cos = set(cos_by_task.keys())
    tasks_with_forgetting = set(forgetting.keys())
    # Also include task 0: it doesn't have cos_sim data during its own training,
    # but we can check: does higher conflict DURING TASK 1 correlate with task 0's forgetting?
    # Better: use task t's cos_sim during task t+1's training vs task t's forgetting.

    # Simple approach: correlate mean cos_sim during task t training vs
    # forgetting of ALL previous tasks combined at the end
    # Or: correlate across tasks (t vs f(t))
    correlation_tasks = sorted(tasks_with_cos & tasks_with_forgetting)

    if verbose:
        print(f"\n--- Correlation analysis ---")
        print(f"  Tasks with both cos_sim & forgetting data: {correlation_tasks}")

    r_direct = float('nan')
    if len(correlation_tasks) >= 2:
        mean_cos = [sum(cos_by_task[t]) / len(cos_by_task[t]) for t in correlation_tasks]
        forget_vals = [forgetting[t] for t in correlation_tasks]
        r_direct = pearson_r(mean_cos, forget_vals)
        if verbose:
            print(f"  r(mean_cos_sim_during_task, task_forgetting) = {r_direct:.4f}")
            if r_direct < -0.3:
                print(f"  => SUPPORTED: negative cos_sim → more forgetting (r={r_direct:.4f})")
            elif abs(r_direct) < 0.1:
                print(f"  => NOT SUPPORTED: uncorrelated (r={r_direct:.4f})")
            else:
                print(f"  => WEAK/AMBIGUOUS (r={r_direct:.4f})")

    # Also: overall cos_sim during task t+1 training vs forgetting of task t
    # (the causal story: conflict while learning new task = forgetting of prev task)
    if verbose:
        print("\n  Cross-task: cos_sim(during t) vs forgetting(t-1)")
    cross_r = float('nan')
    cross_pairs = []
    for t in sorted(cos_by_task.keys()):
        prev_t = t - 1
        if prev_t in forgetting and t in cos_by_task:
            mean_c = sum(cos_by_task[t]) / len(cos_by_task[t])
            f = forgetting[prev_t]
            cross_pairs.append((t, mean_c, f))
            if verbose:
                print(f"    task={t}: mean_cos={mean_c:+.4f}  forgetting(task {prev_t})={f:+.2f}%")
    if len(cross_pairs) >= 2:
        cross_cos = [p[1] for p in cross_pairs]
        cross_forget = [p[2] for p in cross_pairs]
        cross_r = pearson_r(cross_cos, cross_forget)
        if verbose:
            print(f"  r(cos_sim_during_t, forgetting_of_(t-1)) = {cross_r:.4f}")

    # Save CSV
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task', 'n_checks', 'mean_cos_sim', 'std_cos_sim', 'neg_frac',
                       'peak_acc', 'final_acc', 'forgetting'])
            all_tasks = sorted(set(cos_by_task.keys()) | set(peak_acc.keys()))
            for t in all_tasks:
                vals = cos_by_task.get(t, [])
                n = len(vals)
                mean_c = sum(vals) / n if n > 0 else ''
                if n > 1:
                    var = sum((v - mean_c) ** 2 for v in vals) / (n - 1)
                    std_c = var ** 0.5
                else:
                    std_c = ''
                neg_frac = sum(1 for v in vals if v < 0) / n if n > 0 else ''
                w.writerow([
                    t, n,
                    f"{mean_c:.4f}" if mean_c != '' else '',
                    f"{std_c:.4f}" if std_c != '' else '',
                    f"{neg_frac:.3f}" if neg_frac != '' else '',
                    f"{peak_acc.get(t, ''):.2f}" if t in peak_acc else '',
                    f"{final_acc.get(t, ''):.2f}" if t in final_acc else '',
                    f"{forgetting.get(t, ''):.2f}" if t in forgetting else '',
                ])
        if verbose:
            print(f"\n  Saved to {out_path}")

    return {
        'cos_by_task': dict(cos_by_task),
        'peak_acc': peak_acc,
        'final_acc': final_acc,
        'forgetting': forgetting,
        'r_direct': r_direct,
        'cross_r': cross_r,
        'accuracy_snapshots': accuracy_snapshots,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    analyze(args.log, args.out)


if __name__ == '__main__':
    main()

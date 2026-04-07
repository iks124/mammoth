#!/usr/bin/env python3
"""
Parse results for the teleportation experiment.

Target runs (all on seq-cifar10, class-il):
  GPU4/5: er (buffer=500, lr=0.1), derpp_star/er_star (buffer=100, model_config=best)
  GPU6/7: er_tricks (buffer=500, model_config=best), er_ace_star/xder_rpc_star (buffer=100, model_config=best)
  Each model has teleport=0 and teleport=1 (teleport_steps=50) variants.

Metrics reported:
  - Forgetting matrix: accuracy on task i after training up to task j
  - Final (Task 5): standard class-IL metric, comparable to reproduce.json
  - AIA: average of Task 1..5 means (average incremental accuracy)
"""

import os
import json
import numpy as np
import torch
from torch import device  # needed by eval() on logs.pyd

BASE_DIR = '/root/project/mammoth/data/results/class-il/seq-cifar10'
REPRODUCE_JSON = '/root/project/mammoth/scripts/reproduce.json'
N_TASKS = 5

TARGET_CONFIGS = [
    {'model': 'er',           'buffer_size': 500, 'lr': 0.1,  'model_config': None,   'teleport': 0, 'teleport_steps': 50},
    {'model': 'er',           'buffer_size': 500, 'lr': 0.1,  'model_config': None,   'teleport': 1, 'teleport_steps': 50},
    {'model': 'derpp_star',   'buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 0, 'teleport_steps': 50},
    {'model': 'derpp_star',   'buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 1, 'teleport_steps': 50},
    {'model': 'er_star',      'buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 0, 'teleport_steps': 50},
    {'model': 'er_star',      'buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 1, 'teleport_steps': 50},
    {'model': 'er_tricks',    'buffer_size': 500, 'lr': None, 'model_config': 'best', 'teleport': 0, 'teleport_steps': 50},
    {'model': 'er_tricks',    'buffer_size': 500, 'lr': None, 'model_config': 'best', 'teleport': 1, 'teleport_steps': 50},
    {'model': 'er_ace_star',  'buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 0, 'teleport_steps': 50},
    {'model': 'er_ace_star',  'buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 1, 'teleport_steps': 50},
    {'model': 'xder_rpc_star','buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 0, 'teleport_steps': 50},
    {'model': 'xder_rpc_star','buffer_size': 100, 'lr': None, 'model_config': 'best', 'teleport': 1, 'teleport_steps': 50},
]


def load_reproduce_json():
    with open(REPRODUCE_JSON) as f:
        data = json.load(f)
    return {e['model']: e['result'] for e in data if e.get('setting') == 'class-il'}


def parse_logs_pyd(path):
    runs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(eval(line))
            except Exception as e:
                print(f"  [warn] {path}: {e}")
    return runs


def matches(run, cfg):
    if run.get('buffer_size') != cfg['buffer_size']:
        return False
    if cfg['lr'] is not None and run.get('lr') != cfg['lr']:
        return False
    if cfg['model_config'] is not None and run.get('model_config') != cfg['model_config']:
        return False
    if run.get('teleport') != cfg['teleport']:
        return False
    if run.get('teleport_steps') != cfg['teleport_steps']:
        return False
    return True


def extract_metrics(run):
    """
    Returns:
      per_task_means: list[N_TASKS] — mean acc across seen tasks after training task t
      raw_matrix:     list[N_TASKS x N_TASKS] — raw accuracy_i_taskj (None if not applicable)
    """
    raw_matrix = []
    per_task_means = []
    for trained_up_to in range(1, N_TASKS + 1):
        row = []
        for task in range(1, N_TASKS + 1):
            key = f'accuracy_{task}_task{trained_up_to}'
            row.append(run[key] if key in run else None)
        raw_matrix.append(row)
        seen = [v for v in row if v is not None]
        per_task_means.append(np.mean(seen) if seen else None)
    return per_task_means, raw_matrix


def main():
    reproduce = load_reproduce_json()

    header = (f"{'方法':<14} {'T':<2} {'Buffer':<7} {'LR':<6} "
              f"{'任务1':>6} {'任务2':>6} {'任务3':>6} {'任务4':>6} {'最终':>6} {'AIA':>7} {'论文':>7}")
    sep = '-' * (len(header) + 4)
    print(header)
    print(sep)

    results = []
    missing = []

    for cfg in TARGET_CONFIGS:
        model = cfg['model']
        logs_path = os.path.join(BASE_DIR, model, 'logs.pyd')
        if not os.path.exists(logs_path):
            missing.append(cfg)
            continue

        runs = parse_logs_pyd(logs_path)
        matched = [r for r in runs if matches(r, cfg)]
        if not matched:
            missing.append(cfg)
            continue

        run = matched[-1]
        per_task, raw_matrix = extract_metrics(run)

        valid = [x for x in per_task if x is not None]
        if not valid:
            missing.append(cfg)
            continue

        final_acc = per_task[-1] if per_task[-1] is not None else float('nan')
        aia = np.mean(valid)
        actual_lr = run.get('lr', 'N/A')
        paper = reproduce.get(model, 'N/A')

        task_strs = [f"{x:6.2f}" if x is not None else f"{'N/A':>6}" for x in per_task]
        print(f"{model:<14} {cfg['teleport']:<2} {cfg['buffer_size']:<7} {actual_lr:<6} "
              f"{'  '.join(task_strs)}  {aia:6.2f}  {str(paper):>6}")

        results.append({
            'model': model,
            'teleport': cfg['teleport'],
            'buffer_size': cfg['buffer_size'],
            'lr': actual_lr,
            'model_config': run.get('model_config', 'N/A'),
            'per_task': per_task,
            'raw_matrix': raw_matrix,
            'final_acc': final_acc,
            'aia': aia,
            'paper': paper,
        })

    print(sep)
    if missing:
        print(f"\n[未找到 {len(missing)} 个配置]:")
        for cfg in missing:
            print(f"  {cfg['model']} teleport={cfg['teleport']}")

    generate_markdown(results, reproduce)


def fmt(v):
    return f"{v:.1f}" if v is not None else "-"


def generate_markdown(results, reproduce):
    lines = []
    lines.append("# 实验结果总结（Teleportation for CL）\n")
    lines.append('> "最终"列 = 标准 class-IL 最终精度，可与 reproduce.json 直接比较。'
                 '"AIA" = 任务1-5均值快照的平均（平均增量精度），天然偏高，不可与论文值比较。\n')

    # ── Section 1: summary table ──────────────────────────────────────────────
    lines.append("## 汇总\n")
    lines.append("| 方法 | T | Buffer | LR | 任务1 | 任务2 | 任务3 | 任务4 | 最终 | AIA | 论文值 |")
    lines.append("|------|---|--------|----|-------|-------|-------|-------|------|-----|--------|")
    for r in results:
        pt = r['per_task']
        ts = [f"{x:.2f}" if x is not None else "N/A" for x in pt]
        while len(ts) < 5:
            ts.append("N/A")
        lines.append(
            f"| {r['model']} | {r['teleport']} | {r['buffer_size']} | {r['lr']} | "
            f"{ts[0]} | {ts[1]} | {ts[2]} | {ts[3]} | **{r['final_acc']:.2f}** | "
            f"{r['aia']:.2f} | {r['paper']} |"
        )

    # ── Section 2: teleport effect ────────────────────────────────────────────
    lines.append("")
    lines.append("## Teleport 效果对比\n")
    lines.append("| 方法 | 无Teleport(最终) | 有Teleport(最终) | 差值 | 论文基线 | vs论文 |")
    lines.append("|------|-----------------|-----------------|------|---------|--------|")
    models_seen = list(dict.fromkeys(r['model'] for r in results))
    for model in models_seen:
        by_tp = {r['teleport']: r for r in results if r['model'] == model}
        if 0 not in by_tp or 1 not in by_tp:
            continue
        no_tp = by_tp[0]['final_acc']
        with_tp = by_tp[1]['final_acc']
        diff = with_tp - no_tp
        paper = by_tp[0]['paper']
        vs_paper = f"{no_tp - paper:+.2f}%" if isinstance(paper, (int, float)) else "N/A"
        lines.append(f"| {model} | {no_tp:.2f}% | {with_tp:.2f}% | **{diff:+.2f}%** | {paper} | {vs_paper} |")

    # ── Section 3: forgetting matrices ───────────────────────────────────────
    lines.append("")
    lines.append("## 逐任务遗忘矩阵\n")
    lines.append("每个子表：行 = 训练完第N个任务后，列 = 各任务的 class-IL 准确率，最右列为当时所有已见任务的平均值。\n")

    for model in models_seen:
        by_tp = {r['teleport']: r for r in results if r['model'] == model}
        lines.append(f"### {model}\n")

        for tp in [0, 1]:
            if tp not in by_tp:
                continue
            r = by_tp[tp]
            label = "无 Teleport" if tp == 0 else f"有 Teleport (steps={50})"
            lines.append(f"**{label}**\n")

            # table header
            lines.append("| 训练至 | 任务1 | 任务2 | 任务3 | 任务4 | 任务5 | 当前avg |")
            lines.append("|--------|-------|-------|-------|-------|-------|---------|")

            for trained_up_to in range(N_TASKS):
                row = r['raw_matrix'][trained_up_to]
                cells = []
                for task_idx in range(N_TASKS):
                    v = row[task_idx]
                    if v is None:
                        cells.append("—")
                    else:
                        cells.append(f"{v:.1f}")
                avg = r['per_task'][trained_up_to]
                avg_str = f"{avg:.1f}" if avg is not None else "—"
                lines.append(f"| 任务{trained_up_to+1} | {' | '.join(cells)} | {avg_str} |")

            lines.append("")

        # side-by-side forgetting delta if both available
        if 0 in by_tp and 1 in by_tp:
            lines.append(f"**遗忘对比（有Teleport - 无Teleport，正=改善）**\n")
            lines.append("| 训练至 | 任务1 | 任务2 | 任务3 | 任务4 | 任务5 | avg变化 |")
            lines.append("|--------|-------|-------|-------|-------|-------|---------|")
            r0, r1 = by_tp[0], by_tp[1]
            for trained_up_to in range(N_TASKS):
                cells = []
                for task_idx in range(N_TASKS):
                    v0 = r0['raw_matrix'][trained_up_to][task_idx]
                    v1 = r1['raw_matrix'][trained_up_to][task_idx]
                    if v0 is None and v1 is None:
                        cells.append("—")
                    elif v0 is None or v1 is None:
                        cells.append("?")
                    else:
                        d = v1 - v0
                        cells.append(f"**{d:+.1f}**" if abs(d) >= 3 else f"{d:+.1f}")
                pt0 = r0['per_task'][trained_up_to]
                pt1 = r1['per_task'][trained_up_to]
                avg_d = (pt1 - pt0) if (pt0 is not None and pt1 is not None) else None
                avg_str = (f"**{avg_d:+.1f}**" if avg_d is not None and abs(avg_d) >= 3
                           else f"{avg_d:+.1f}" if avg_d is not None else "—")
                lines.append(f"| 任务{trained_up_to+1} | {' | '.join(cells)} | {avg_str} |")
            lines.append("")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_summary.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()

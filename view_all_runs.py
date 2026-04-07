#!/usr/bin/env python3

import os
import torch
import numpy as np
from torch import device

def parse_logs_pyd(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.strip().split('\n')
    all_runs = []
    for line in lines:
        if not line.strip():
            continue
        try:
            log_dict = eval(line.strip())
            all_runs.append(log_dict)
        except Exception as e:
            print(f"Error parsing line in {path}: {e}")
            continue
    return all_runs

def get_results_per_method(method_dir):
    logs_path = os.path.join(method_dir, 'logs.pyd')
    if not os.path.exists(logs_path):
        print(f"Warning: {logs_path} does not exist.")
        return []
    return parse_logs_pyd(logs_path)

def extract_task_accuracies(run, n_tasks=5):
    task_accuracies = []
    for trained_up_to in range(1, n_tasks + 1):
        task_accs = []
        for task in range(1, trained_up_to + 1):
            key = f'accuracy_{task}_task{trained_up_to}'
            if key in run:
                task_accs.append(run[key])
        task_accuracies.append(task_accs)
    return task_accuracies

def main():
    base_dir = '/root/project/mammoth/data/results/class-il/seq-cifar10'
    methods = sorted(os.listdir(base_dir))
    methods = [m for m in methods if os.path.isdir(os.path.join(base_dir, m))]

    for method in methods:
        print(f"\n=== Method: {method} ===")
        method_dir = os.path.join(base_dir, method)
        runs = get_results_per_method(method_dir)

        if len(runs) == 0:
            print("No valid runs found.")
            continue

        print(f"Found {len(runs)} runs:")
        print('-' * 80)
        print(f"{'Run':<6} {'Seed':<8} {'Buffer':<8} {'LR':<10} {'Avg Task 1':<10} {'Avg Task 5':<10} {'Total Avg':<10}")
        print('-' * 80)

        run_stats = []

        for i, run in enumerate(runs):
            seed = run.get('seed', 'N/A')
            buffer_size = run.get('buffer_size', 'N/A')
            lr = run.get('lr', 'N/A')
            task_accuracies = extract_task_accuracies(run)
            task_accs = []
            for t in range(len(task_accuracies)):
                task_avg = sum(task_accuracies[t]) / len(task_accuracies[t])
                task_accs.append(task_avg)
            avg_total = sum(task_accs) / len(task_accs)
            avg_task1 = task_accs[0] if task_accs else 0
            avg_task5 = task_accs[-1] if task_accs else 0

            run_stats.append({
                'run': i + 1,
                'seed': seed,
                'buffer': buffer_size,
                'lr': lr,
                'avg_task1': avg_task1,
                'avg_task5': avg_task5,
                'total_avg': avg_total
            })

        for stat in run_stats:
            seed_str = str(stat['seed']) if stat['seed'] is not None else 'N/A'
            buffer_str = str(stat['buffer']) if stat['buffer'] is not None else 'N/A'
            lr_str = str(stat['lr']) if stat['lr'] is not None else 'N/A'
            print(f"{stat['run']:<6} {seed_str:<8} {buffer_str:<8} {lr_str:<10} "
                  f"{stat['avg_task1']:<10.2f} {stat['avg_task5']:<10.2f} {stat['total_avg']:<10.2f}")

        if len(run_stats) > 1:
            valid_runs = [stat for stat in run_stats if stat['total_avg'] > 10]
            if valid_runs:
                avg_total = np.mean([stat['total_avg'] for stat in valid_runs])
                avg_task1 = np.mean([stat['avg_task1'] for stat in valid_runs])
                avg_task5 = np.mean([stat['avg_task5'] for stat in valid_runs])
                print('-' * 80)
                print(f"{'Avg':<6} {'-':<8} {'-':<8} {'-':<10} "
                      f"{avg_task1:<10.2f} {avg_task5:<10.2f} {avg_total:<10.2f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Script to parse and summarize results from Mammoth experiments stored in logs.pyd files.
"""

import os
import ast
import numpy as np

import torch
import numpy as np
from torch import device

def parse_logs_pyd(path):
    """
    Parse logs.pyd file.
    """
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
    """
    Get all results for a specific method.
    """
    logs_path = os.path.join(method_dir, 'logs.pyd')
    if not os.path.exists(logs_path):
        print(f"Warning: {logs_path} does not exist.")
        return []

    return parse_logs_pyd(logs_path)

def main():
    # Base directory containing all method subdirectories
    base_dir = '/root/project/mammoth/data/results/class-il/seq-cifar10'

    # List all method subdirectories
    methods = sorted(os.listdir(base_dir))
    methods = [m for m in methods if os.path.isdir(os.path.join(base_dir, m))]

    print(f"{'Method':<20} {'Task 1':<8} {'Task 2':<8} {'Task 3':<8} {'Task 4':<8} {'Task 5':<8} {'Average':<8}")
    print('-' * 80)

    method_avg_accs = []

    for method in methods:
        method_dir = os.path.join(base_dir, method)
        runs = get_results_per_method(method_dir)

        if len(runs) == 0:
            print(f"{method:<20} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
            continue

        # For each run, get the task accuracies (class-il setting) for each task
        task_accs = []

        for run in runs:
            # Extract task accuracies from accuracy_X_taskY fields
            task_acc_per_run = []
            for trained_up_to in range(1, 6):
                accs = []
                for task in range(1, trained_up_to + 1):
                    key = f'accuracy_{task}_task{trained_up_to}'
                    if key in run:
                        accs.append(run[key])
                task_acc_per_run.append(accs)
            task_accs.append(task_acc_per_run)

        # Convert to numpy array for averaging
        max_len = max(len(run) for run in task_accs)
        padded_task_accs = []
        for run in task_accs:
            padded_run = []
            for task in range(max_len):
                if task < len(run):
                    padded_run.append(np.array(run[task]))
                else:
                    padded_run.append(np.array([]))
            padded_task_accs.append(padded_run)
        task_accs = np.array(padded_task_accs, dtype=object)

        # Average over all runs
        avg_accs_per_task = []
        for task in range(max_len):
            task_runs = [run[task] for run in task_accs if len(run[task]) > 0]
            if task_runs:
                avg_acc = np.mean([np.mean(run) for run in task_runs])
                avg_accs_per_task.append(avg_acc)
            else:
                avg_accs_per_task.append(np.nan)

        # Collect average accuracy after each task
        method_avg = []
        for t in range(len(avg_accs_per_task)):
            task_avg = avg_accs_per_task[t]
            method_avg.append(task_avg)

        method_avg_accs.append((method, method_avg))

        # Print results for the method
        method_str = method
        task_strs = []

        for t in range(len(avg_accs_per_task)):
            acc = avg_accs_per_task[t]
            task_strs.append(f"{acc:.2f}")

        avg_total = np.nanmean(method_avg)
        task_strs.append(f"{avg_total:.2f}")

        print(f"{method_str:<20} {' '.join([f'{s:<8}' for s in task_strs])}")

    print('-' * 80)

if __name__ == "__main__":
    main()

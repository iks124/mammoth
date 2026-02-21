from copy import deepcopy
from typing import Dict

import torch
from tqdm.auto import tqdm

from models.clip_ft_utils.ties_merging import ties_merging


def add_merging_args(parser):
    parser.add_argument(
        "--merging",
        type=str,
        default="ta",
        choices=["ta", "dare", "iso", "ties", "tsv", "tsv_core"],
    )
    parser.add_argument("--alpha_merging", type=float, default=1.0)


def get_merging_function(command_args, device):
    if command_args.merging == "ta":
        return TaskArithmetic(device, alpha=command_args.alpha_merging)
    if command_args.merging == "dare":
        return DARE(device, alpha=command_args.alpha_merging)
    if command_args.merging == "iso":
        return ISO(device, alpha=command_args.alpha_merging)
    if command_args.merging == "ties":
        return TIES(device, alpha=command_args.alpha_merging)
    if command_args.merging == "tsv":
        return TSV(device, alpha=command_args.alpha_merging)
    if command_args.merging == "tsv_core":
        return TSV_CORE(device, alpha=command_args.alpha_merging)
    raise ValueError


class AbstractMerging:
    def merge(self):
        raise NotImplementedError

    def add(self: Dict):
        raise NotImplementedError

    def set_alpha(self, alpha: float):
        self.alpha = alpha


class TaskArithmetic(AbstractMerging):
    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self.scaled_sum: Dict = None

    @torch.no_grad()
    def merge(self, names=None):
        alpha = (1 / self.num_tasks) * self.alpha
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k in self._running_sum.keys():
            self.scaled_sum[k].copy_(self._running_sum[k])
            self.scaled_sum[k].mul_(alpha)
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            self._running_sum[k].add_(v)
        self.num_tasks += 1


class DARE(AbstractMerging):
    def __init__(self, device, alpha: float = 1.0, p: float = 0.7):
        self.device = device
        self.alpha = alpha
        self.p = p
        self.num_tasks = 0
        self._running_sum: Dict = None
        self.scaled_sum: Dict = None

    def randbin(self, m, n):
        return (
            torch.randint(2, size=(m, n), dtype=torch.float32)
            .bernoulli(1 - self.p)
            .to(self.device)
        )

    @torch.no_grad()
    def merge(self, names=None):
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k, v in self._running_sum.items():
            self.scaled_sum[k].copy_(v)
            self.scaled_sum[k].mul_(self.alpha * (1 / self.num_tasks))
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            if len(v.shape) != 2:
                self._running_sum[k].add_(v)
            else:
                mask_ = self.randbin(v.shape[0], v.shape[1])
                self._running_sum[k].add_(v * mask_ * (1 / (1 - self.p)))
        self.num_tasks += 1


class ISO(AbstractMerging):
    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self.scaled_sum: Dict = None

    @torch.no_grad()
    def merge(self, names=None):
        assert self._running_sum.keys() == self.scaled_sum.keys()
        for k, v in tqdm(
            self._running_sum.items(),
            desc="Computing SVD for ISO",
            total=len(self._running_sum),
        ):
            self.scaled_sum[k].copy_(v)
            self.scaled_sum[k].div_(self.num_tasks)
            if len(v.shape) == 2:
                u, s, vh = torch.linalg.svd(
                    self.scaled_sum[k].to(torch.double), full_matrices=False
                )
                self.scaled_sum[k].copy_((u @ vh).to(self.scaled_sum[k].dtype))
                self.scaled_sum[k].mul_(s.mean())
            self.scaled_sum[k].mul_(self.alpha)
        if names is None:
            return self.scaled_sum
        assert self.scaled_sum.keys() == set(names)
        return [self.scaled_sum[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self._running_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self.scaled_sum = {k: torch.zeros_like(v) for k, v in param_dict.items()}
        assert param_dict.keys() == self._running_sum.keys()
        for k, v in param_dict.items():
            self._running_sum[k].add_(v)
        self.num_tasks += 1


class TIES(AbstractMerging):
    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self._separated_task_vectors: Dict = None
        self.merged_model: Dict = None

    def apply_ta(self, v):
        return len(v.shape) != 2

    @torch.no_grad()
    def merge(self, names=None):
        for k, v in self._running_sum.items():
            self.merged_model[k].copy_(v)
            self.merged_model[k].mul_(self.alpha / self.num_tasks)
        for k, v in self._separated_task_vectors.items():
            merged_tv, _, _ = ties_merging(v)
            self.merged_model[k].copy_((self.alpha / self.num_tasks) * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._running_sum = {
                k: torch.zeros_like(v)
                for k, v in param_dict.items()
                if self.apply_ta(v)
            }
            self._separated_task_vectors = {
                k: [] for k, v in param_dict.items() if not self.apply_ta(v)
            }
        for k, v in param_dict.items():
            if self.apply_ta(v):
                self._running_sum[k].add_(v)
            else:
                self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1


class TSV(AbstractMerging):
    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self._separated_task_vectors: Dict = None
        self.merged_model: Dict = None

    def apply_ta(self, v):
        return len(v.shape) != 2

    @torch.no_grad()
    def get_tsv_delta_w(self, ftms_task_dirs):
        sv_reduction = 1 / len(ftms_task_dirs)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, vh = torch.linalg.svd(vec.to(torch.float64), full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(vh)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = vh[
                :reduced_index_s, :
            ]
        u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).type_as(
            ftms_task_dirs[0]
        )

    @torch.no_grad()
    def merge(self, names=None):
        for k, v in self._running_sum.items():
            self.merged_model[k].copy_(v)
            self.merged_model[k].div_(self.num_tasks)
            self.merged_model[k].mul_(self.alpha)
        for k, v in self._separated_task_vectors.items():
            merged_tv = self.get_tsv_delta_w(v)
            merged_tv = (
                merged_tv.type_as(v[0]) if hasattr(merged_tv, "type_as") else merged_tv
            )
            self.merged_model[k].copy_(self.alpha * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._running_sum = {
                k: torch.zeros_like(v)
                for k, v in param_dict.items()
                if self.apply_ta(v)
            }
            self._separated_task_vectors = {
                k: [] for k, v in param_dict.items() if not self.apply_ta(v)
            }
        for k, v in param_dict.items():
            if self.apply_ta(v):
                self._running_sum[k].add_(v)
            else:
                self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1


class TSV_CORE(AbstractMerging):
    def __init__(self, device, alpha: float = 1.0):
        self.device = device
        self.alpha = alpha
        self.num_tasks = 0
        self._running_sum: Dict = None
        self._separated_task_vectors: Dict = None
        self.merged_model: Dict = None

    @torch.no_grad()
    def get_tsv_delta_w(self, ftms_task_dirs):
        sv_reduction = 1 / len(ftms_task_dirs)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, vh = torch.linalg.svd(vec.to(torch.float64), full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(vh)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = vh[
                :reduced_index_s, :
            ]
        u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).type_as(
            ftms_task_dirs[0]
        )

    @torch.no_grad()
    def merge(self, names=None):
        for k, v in self._separated_task_vectors.items():
            a_list = [x[0] for x in v]
            b_list = [x[1] for x in v]
            a_stack = torch.cat(a_list, dim=0)
            b_stack = torch.cat(b_list, dim=1)

            vh_a_ref = torch.linalg.svd(a_stack.to(torch.float64), full_matrices=False)[
                2
            ]
            u_b_ref = torch.linalg.svd(b_stack.to(torch.float64), full_matrices=False)[
                0
            ]

            m_list = []
            for a, b in zip(a_list, b_list):
                m_aligned = (u_b_ref.T @ b) @ (a @ vh_a_ref.T)
                m_list.append(m_aligned)

            merged_tv = self.get_tsv_delta_w(m_list)
            merged_tv = (
                merged_tv.type_as(v[0]) if hasattr(merged_tv, "type_as") else merged_tv
            )
            self.merged_model[k].copy_(self.alpha * merged_tv)
        if names is None:
            return self.merged_model
        assert self.merged_model.keys() == set(names)
        return [self.merged_model[n] for n in names]

    @torch.no_grad()
    def add(self, param_dict: Dict):
        if self._running_sum is None:
            self.merged_model = {k: torch.zeros_like(v) for k, v in param_dict.items()}
            self._separated_task_vectors = {k: [] for k, _ in param_dict.items()}
        for k, v in param_dict.items():
            self._separated_task_vectors[k].append(torch.clone(v))
        self.num_tasks += 1

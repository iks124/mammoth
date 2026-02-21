import math
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

from utils import binary_to_boolean_type


def set_requires_grad_to(model, namevars, mode: bool):
    for n, p in model.named_parameters():
        if n in namevars:
            p.requires_grad = mode


def add_clip_args(parser):
    parser.add_argument(
        "--clip_backbone",
        type=str,
        default="ViT-B/16",
        choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"],
        help="Backbone architecture for CLIP",
    )
    parser.add_argument("--ft_linears", type=binary_to_boolean_type, default=1)
    parser.add_argument("--ft_attention", type=binary_to_boolean_type, default=1)
    parser.add_argument("--ft_ln", type=binary_to_boolean_type, default=1)
    parser.add_argument("--ft_class_embed", type=binary_to_boolean_type, default=1)
    parser.add_argument("--ft_proj", type=binary_to_boolean_type, default=1)
    parser.add_argument("--ft_pos_embed", type=binary_to_boolean_type, default=0)
    parser.add_argument("--ft_conv", type=binary_to_boolean_type, default=0)


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    if not isinstance(min_lr, list):
        min_lr_list = [min_lr for _ in optimizer.param_groups]
    else:
        min_lr_list = min_lr
    assert len(base_lrs) == len(optimizer.param_groups) == len(min_lr_list)

    def _lr_adjuster(step):
        for param_group, base_lr, group_min_lr in zip(
            optimizer.param_groups, base_lrs, min_lr_list
        ):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = max(1, steps - warmup_length)
                lr = group_min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def step_lr_decay(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                progress = step / max(1, steps)
                if progress < 0.70:
                    lr = base_lr
                elif progress < 0.90:
                    lr = base_lr * 0.5
                else:
                    lr = base_lr * 0.1
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


class OptimizerBuilder:
    def __init__(self, cmd_args):
        self.args = cmd_args

    def build_opt_and_sched(self, all_params, num_batches):
        if self.args.optimizer == "adamw":
            opt = torch.optim.AdamW(
                all_params, lr=self.args.lr, weight_decay=self.args.optim_wd
            )
        elif self.args.optimizer == "sgd":
            opt = torch.optim.SGD(
                all_params,
                lr=self.args.lr,
                momentum=self.args.optim_mom,
                weight_decay=self.args.optim_wd,
            )
        else:
            raise ValueError

        sched = None
        if self.args.scheduler_ntk == "none":
            pass
        elif self.args.scheduler_ntk == "cosine":
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            sched = cosine_lr(opt, self.args.lr, 500, num_total_steps, 0)
        elif self.args.scheduler_ntk == "cosine_plus":
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            warmup_steps = int(0.1 * num_total_steps)
            sched = cosine_lr(
                opt, self.args.lr, warmup_steps, num_total_steps, 0.1 * self.args.lr
            )
        elif self.args.scheduler_ntk == "decay":
            sched = cosine_lr(opt, self.args.lr, 0, self.args.n_epochs * num_batches, 0)
        elif self.args.scheduler_ntk == "step":
            num_steps = self.args.n_epochs * num_batches // self.args.chunks
            warmup_steps = int(0.1 * num_steps)
            sched = step_lr_decay(opt, self.args.lr, warmup_steps, num_steps)
        else:
            raise ValueError

        return opt, sched


@torch.no_grad()
def compute_acc_on_last_task(model, dataset):
    test_loader = dataset.test_loaders[-1]
    total_len = len(test_loader) if hasattr(test_loader, "__len__") else None
    pbar = tqdm(
        test_loader, total=total_len, desc="Evaluating", disable=model.args.non_verbose
    )

    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    test_iter = iter(test_loader)
    i = 0
    num_classes = dataset.N_CLASSES
    while True:
        try:
            data = next(test_iter)
        except StopIteration:
            break
        if model.args.debug_mode and i > model.get_debug_iters():
            break
        inputs, labels = data[0], data[1]
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model.forward(inputs)
        assert outputs.shape[1] == num_classes
        _, pred = torch.max(outputs, 1)
        correct += torch.sum(pred == labels).item()
        total += labels.shape[0]
        i += 1
        pbar.set_postfix(
            {f"acc_task_{model.current_task + 1}": max(0, correct / total * 100)},
            refresh=False,
        )
        pbar.update(1)

        start_c, end_c = dataset.get_offsets(model.current_task)
        outputs[:, :start_c] = -float("inf")
        outputs[:, end_c:num_classes] = -float("inf")
        _, pred = torch.max(outputs.data, 1)
        correct_mask_classes += torch.sum(pred == labels).item()

    pbar.close()
    return (correct / total * 100), (correct_mask_classes / total * 100)


class StubCompressor:
    def __call__(self, x):
        return x


class HalfCompressor:
    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.half().to(orig_dtype)
        x = 0.5 * (x + x.t())
        return x


class SVDCompressor:
    def __init__(self, rank=32):
        self.rank = rank

    def __call__(self, x):
        orig_dtype = x.dtype
        x64 = x.to(torch.float64)
        u, s, vh = torch.linalg.svd(x64, full_matrices=False)
        r_max = s.shape[0]
        if isinstance(self.rank, float) and 0 < self.rank < 1:
            k = max(1, int(r_max * self.rank))
        elif isinstance(self.rank, int):
            k = min(self.rank, r_max)
        else:
            raise ValueError
        u_k = u[:, :k]
        s_k = s[:k]
        vh_k = vh[:k, :]
        x_approx = (u_k * s_k) @ vh_k
        x_approx = 0.5 * (x_approx + x_approx.t())
        return x_approx.to(orig_dtype)


class DiagonalCompressor:
    def __call__(self, x):
        return torch.diag(torch.diag(x))


class DynamicMatrixCompressor:
    def __init__(self, bits=8):
        assert bits in (8, 16)
        self.bits = bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if self.bits == 8:
            qmin, qmax, qdtype = -128, 127, torch.int8
        else:
            qmin, qmax, qdtype = -32768, 32767, torch.int16

        min_val = torch.quantile(x, 0.03)
        max_val = torch.quantile(x, 0.97)
        if max_val == min_val:
            return x.clone().to(orig_dtype)

        x_clamped = torch.clamp(x, min_val, max_val)
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        q_x = torch.round(x_clamped / scale + zero_point)
        q_x = torch.clamp(q_x, qmin, qmax).to(qdtype)
        x_q = (q_x.to(torch.float32) - zero_point) * scale
        x_q = 0.5 * (x_q + x_q.t())
        return x_q.to(orig_dtype)


class ThresholdCompressor:
    def __init__(self, keep_ratio: float):
        assert 0 < keep_ratio <= 1
        self.keep_ratio = keep_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[0] == x.shape[1]
        n = x.shape[0]
        iu, ju = torch.triu_indices(n, n, offset=0, device=x.device)
        tri = x[iu, ju]
        tri_abs = tri.abs()
        k = int(self.keep_ratio * tri_abs.numel())
        if k <= 0:
            x_c = torch.zeros_like(x)
            diag = torch.diag(x)
            idx = torch.arange(n, device=x.device)
            x_c[idx, idx] = diag
            return x_c
        topk_vals, _ = torch.topk(tri_abs, k, sorted=True)
        tau = topk_vals[-1].item()
        tri_pruned = tri.clone()
        tri_pruned[tri_abs < tau] = 0.0
        x_c = torch.zeros_like(x)
        x_c[iu, ju] = tri_pruned
        il, jl = torch.tril_indices(n, n, offset=-1, device=x.device)
        x_c[il, jl] = x_c[jl, il]
        return x_c


class BlockDiagonalCompressor:
    def __init__(self, num_blocks: int = 8, adapt_blocks: bool = True):
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.adapt_blocks = adapt_blocks

    def _nearest_divisor(self, n: int) -> int:
        target = min(self.num_blocks, n)
        best = 1
        best_dist = abs(target - 1)
        for d in range(2, n + 1):
            if n % d == 0:
                dist = abs(target - d)
                if dist < best_dist:
                    best = d
                    best_dist = dist
        return best

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[0] == x.shape[1]
        n = x.shape[0]
        num_blocks = (
            self._nearest_divisor(n) if self.adapt_blocks else min(self.num_blocks, n)
        )
        block_size = n // num_blocks
        x_c = torch.zeros_like(x)
        for b in range(num_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < num_blocks - 1 else n
            x_c[start:end, start:end] = x[start:end, start:end]
        return x_c


class FisherLoader:
    def __init__(
        self, fisher_cache, dataset_name, device, precision="fp32", compressor=None
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.fisher_cache = fisher_cache
        self.precision = precision

        if compressor in (None, "none"):
            self.compressor = StubCompressor()
        elif compressor == "half":
            self.compressor = HalfCompressor()
        elif compressor == "svd":
            self.compressor = SVDCompressor()
        elif compressor == "svd++":
            self.compressor = SVDCompressor(rank=0.25)
        elif compressor == "svd++++":
            self.compressor = SVDCompressor(rank=0.15)
        elif compressor == "diag":
            self.compressor = DiagonalCompressor()
        elif compressor == "dyn8":
            self.compressor = DynamicMatrixCompressor(8)
        elif compressor == "dyn16":
            self.compressor = DynamicMatrixCompressor(16)
        elif compressor == "pruning":
            self.compressor = ThresholdCompressor(0.3)
        elif compressor == "pruning++":
            self.compressor = ThresholdCompressor(0.10)
        elif compressor == "block":
            self.compressor = BlockDiagonalCompressor()
        else:
            raise ValueError

    def load_kfac(self, task_id, only_counts=False):
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        path_num_aa = fisher_cache_path.replace(".pt", "_num_aaT.pt")
        path_num_gg = fisher_cache_path.replace(".pt", "_num_ggT.pt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if os.path.exists(path_num_aa):
                assert os.path.exists(path_num_gg)
                cur_num_aa = torch.load(path_num_aa, map_location="cpu").item()
                cur_num_gg = torch.load(path_num_gg, map_location="cpu").item()
            else:
                raise FileNotFoundError(
                    f"Fisher cache file {path_num_aa} or {path_num_gg} not found."
                )
        if only_counts:
            return cur_num_gg, cur_num_aa

        path_aa = fisher_cache_path.replace(".pt", "_aaT.pt")
        path_gg = fisher_cache_path.replace(".pt", "_ggT.pt")
        path_ff = fisher_cache_path.replace(".pt", "_ffT.pt")
        assert os.path.exists(path_aa)
        assert os.path.exists(path_gg)
        assert os.path.exists(path_ff)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aa_t = torch.load(path_aa, map_location=self.device)
            gg_t = torch.load(path_gg, map_location=self.device)
            ff_t = torch.load(path_ff, map_location=self.device)

        for key in aa_t.keys():
            if self.precision == "fp64":
                aa_t[key] = self.compressor(aa_t[key]).to(torch.float64)
                gg_t[key] = self.compressor(gg_t[key]).to(torch.float64)
            elif self.precision == "fp32":
                aa_t[key] = self.compressor(aa_t[key]).to(torch.float32)
                gg_t[key] = self.compressor(gg_t[key]).to(torch.float32)
            else:
                raise NotImplementedError

        for key in ff_t.keys():
            if self.precision == "fp64":
                ff_t[key] = self.compressor(ff_t[key]).to(torch.float64)
            elif self.precision == "fp32":
                ff_t[key] = self.compressor(ff_t[key]).to(torch.float32)
            else:
                raise NotImplementedError

        return gg_t, aa_t, ff_t, cur_num_gg, cur_num_aa

    def store_kfac(self, task_id, gg_t, aa_t, ff_t, num_gg_t, num_aa_t):
        os.makedirs(self.fisher_cache, exist_ok=True)
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        torch.save(gg_t, fisher_cache_path.replace(".pt", "_ggT.pt"))
        torch.save(aa_t, fisher_cache_path.replace(".pt", "_aaT.pt"))
        torch.save(ff_t, fisher_cache_path.replace(".pt", "_ffT.pt"))
        torch.save(
            torch.tensor([num_gg_t]), fisher_cache_path.replace(".pt", "_num_ggT.pt")
        )
        torch.save(
            torch.tensor([num_aa_t]), fisher_cache_path.replace(".pt", "_num_aaT.pt")
        )


def get_parameter(
    shape,
    device,
    type_init: str = "orto",
    transpose: bool = False,
    requires_grad: bool = True,
):
    param = torch.zeros(*shape, dtype=torch.float32, device=device)
    if type_init == "orto":
        torch.nn.init.orthogonal_(param)
    if type_init == "gaussian":
        torch.nn.init.normal_(param, mean=0.0, std=0.1)
    if type_init == "kernel":
        torch.nn.init.normal_(param, mean=0.0, std=0.036)
    if type_init == "attn":
        torch.nn.init.normal_(param, mean=1.0, std=0.03)
    if type_init == "kaiming":
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    if type_init == "ones":
        torch.nn.init.ones_(param)
    if transpose:
        param = torch.transpose(param, 1, 2)
    return torch.nn.Parameter(param, requires_grad=requires_grad)

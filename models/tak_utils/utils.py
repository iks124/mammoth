import torch
import math
import os
import warnings

from utils import binary_to_boolean_type
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset

from tqdm import tqdm

try:
    import clip
except ImportError:
    raise ImportError("Please install the CLIP package by running: pip install git+https://github.com/openai/CLIP.git")

import numpy as np


def set_requires_grad_to(model, namevars, mode: bool):
    for n, p in model.named_parameters():
        if n in namevars:
            p.requires_grad = mode


def add_clip_args(parser):
    parser.add_argument('--clip_backbone', type=str, default='ViT-B/16',
                        help='Backbone architecture for CLIP', choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14'])
    parser.add_argument('--ft_linears', type=binary_to_boolean_type, default=1,
                        help='Set to 1 fine-tune linear layers')
    parser.add_argument('--ft_attention', type=binary_to_boolean_type, default=1,
                        help='Set to 1 fine-tune attention layers')
    parser.add_argument('--ft_ln', type=binary_to_boolean_type, default=1, help='Set to 1 fine-tune layer norm')
    parser.add_argument('--ft_class_embed', type=binary_to_boolean_type, default=1,
                        help='Set to 1 fine-tune class embedding layers')
    parser.add_argument('--ft_proj', type=binary_to_boolean_type, default=1,
                        help='Set to 1 fine-tune projection layers')

    parser.add_argument('--ft_pos_embed', type=binary_to_boolean_type, default=0,
                        help='Set to 1 fine-tune posistional embedding')
    parser.add_argument('--ft_conv', type=binary_to_boolean_type, default=0,
                        help='Set to 1 fine-tune convolutional layers')


class OptimizerBuilder:

    def __init__(self, cmd_args):
        self.args = cmd_args

    def build_opt_and_sched(self, all_params, num_batches):
        opt, sched = None, None

        if self.args.optimizer == 'adamw':
            opt = torch.optim.AdamW(all_params, lr=self.args.lr, weight_decay=self.args.optim_wd)
        elif self.args.optimizer == 'sgd':
            opt = torch.optim.SGD(all_params, lr=self.args.lr,
                                  momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        else:
            raise ValueError
        
        
        reduction_factor = getattr(self.args, 'epochs_factor_reduction', 1)

        if self.args.scheduler_ntk == 'none':
            pass
        elif self.args.scheduler_ntk == 'cosine':
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            sched = cosine_lr(opt, self.args.lr, 500 / reduction_factor, num_total_steps, 0)
        elif self.args.scheduler_ntk == 'cosine_talos':
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            sched = cosine_lr(opt, self.args.lr, 200, num_total_steps, 0)
        elif self.args.scheduler_ntk == 'cosine_plus':
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            warmup_steps = int(0.1 * num_total_steps)
            sched = cosine_lr(opt, self.args.lr, warmup_steps, num_total_steps, 0.1 * self.args.lr)
        elif self.args.scheduler_ntk == 'decay':
            sched = cosine_lr(opt, self.args.lr, 0, self.args.n_epochs * num_batches, 0)
        elif self.args.scheduler_ntk == 'step':
            num_steps = self.args.n_epochs * num_batches // self.args.chunks
            warmup_steps = int(0.1 * num_steps)

            sched = step_lr_decay(opt, self.args.lr, warmup_steps, num_steps)
        else:
            raise ValueError

        return opt, sched

    
    def build_opt_and_sched_multiple_lr(self, params_group_1, params_group_2, num_batches):
        opt, sched = None, None

        lr_group_1 = self.args.lr
        lr_group_2 = getattr(self.args, 'lr2', None)
        if lr_group_2 is None:
            lr_group_2 = getattr(self.args, 'lr_lin', None)
        if lr_group_2 is None:
            lr_group_2 = getattr(self.args, 'lr_second', None)
        if lr_group_2 is None or lr_group_2 == 0:
            lr_group_2 = self.args.lr

        param_groups = [
            {"params": params_group_1, "lr": lr_group_1},
            {"params": params_group_2, "lr": lr_group_2}
        ]

        if self.args.optimizer == 'adamw':
            opt = torch.optim.AdamW(param_groups, lr=lr_group_1, weight_decay=self.args.optim_wd)
        elif self.args.optimizer == 'sgd':
            opt = torch.optim.SGD(param_groups, lr=lr_group_1,
                                  momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        else:
            raise ValueError

        base_lrs = [lr_group_1, lr_group_2]

        if self.args.scheduler_ntk == 'none':
            pass
        elif self.args.scheduler_ntk == 'cosine':
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            sched = cosine_lr(opt, base_lrs, 500, num_total_steps, 0)
        elif self.args.scheduler_ntk == 'cosine_talos':
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            sched = cosine_lr(opt, base_lrs, 200, num_total_steps, 0)
        elif self.args.scheduler_ntk == 'cosine_plus':
            num_total_steps = self.args.n_epochs * (num_batches // self.args.chunks)
            warmup_steps = int(0.1 * num_total_steps)
            min_lrs = [0.1 * lr for lr in base_lrs]
            sched = cosine_lr(opt, base_lrs, warmup_steps, num_total_steps, min_lrs)
        elif self.args.scheduler_ntk == 'decay':
            sched = cosine_lr(opt, base_lrs, 0, self.args.n_epochs * num_batches, 0)
        elif self.args.scheduler_ntk == 'step':
            num_steps = self.args.n_epochs * num_batches // self.args.chunks
            warmup_steps = int(0.1 * num_steps)

            sched = step_lr_decay(opt, base_lrs, warmup_steps, num_steps)
        else:
            raise ValueError

        return opt, sched


@torch.no_grad()
def compute_acc_on_last_task(model: ContinualModel, dataset: ContinualDataset):
    test_loader = dataset.test_loaders[-1]
    total_len = len(test_loader) if hasattr(test_loader, '__len__') else None

    pbar = tqdm(test_loader, total=total_len,
                desc='Evaluating', disable=model.args.non_verbose)

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
        pbar.set_postfix({f'acc_task_{model.current_task + 1}': max(0, correct / total * 100)}, refresh=False)
        pbar.set_description(f"Evaluating Task {model.current_task + 1}", refresh=False)
        pbar.update(1)

        start_c, end_c = dataset.get_offsets(model.current_task)
        outputs[:, :start_c] = -float('inf')
        outputs[:, end_c:num_classes] = -float('inf')
        _, pred = torch.max(outputs.data, 1)
        correct_mask_classes += torch.sum(pred == labels).item()

    acc = (correct / total * 100)
    acc_mask_classes = (correct_mask_classes / total * 100)

    pbar.close()

    return acc, acc_mask_classes


def make_psd(x, to64 = False):
    orig_dtype = x.dtype
    if to64:
        x = x.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals_clamped = torch.clamp(eigvals, min=0.0)
    x_psd = (eigvecs * eigvals_clamped) @ eigvecs.t()
    return x_psd.to(orig_dtype) if to64 else x_psd


class StubCompressor:

    def __init__(self):
        pass

    def __call__(self, x):
        return x


class HalfCompressor:

    def __init__(self):
        pass

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.half().to(orig_dtype)
        x = 0.5 * (x + x.t())
        #x = make_psd(x, True)
        return x


class SVDCompressor:

    def __init__(self, rank=32):
        self.rank = rank

    def __call__(self, x):
        orig_dtype = x.dtype

        x64 = x.to(torch.float64)
        U, S, Vh = torch.linalg.svd(x64, full_matrices=False)

        r_max = S.shape[0]

        if isinstance(self.rank, float) and 0 < self.rank < 1:
            k = max(1, int(r_max * self.rank))
        elif isinstance(self.rank, int):
            k = min(self.rank, r_max)
        else:
            raise ValueError("rank deve essere un intero oppure una percentuale tra 0 e 1")

        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

        x_approx = (U_k * S_k) @ Vh_k

        x_approx = 0.5 * (x_approx + x_approx.t())

        return x_approx.to(orig_dtype)


class DiagonalCompressor:

    def __init__(self):
        pass

    def __call__(self, x):
        return torch.diag(torch.diag(x))


class DynamicMatrixCompressor:

    def __init__(self, bits=8):
        assert bits in (8, 16), "bits must be 8 or 16"
        self.bits = bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        if self.bits == 8:
            qmin, qmax = -128, 127
            qdtype = torch.int8
        else:
            qmin, qmax = -32768, 32767
            qdtype = torch.int16

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
        x_q = make_psd(x_q)

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

        num_unique = tri_abs.numel()
        k = int(self.keep_ratio * num_unique)
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

        diag_idx = torch.arange(n, device=x.device)
        x_c[diag_idx, diag_idx] = x[diag_idx, diag_idx]

        x_c = make_psd(x_c)

        return x_c


class BlockDiagonalCompressor:
    def __init__(self, num_blocks: int = 8,
                 adapt_blocks: bool = True):
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

        if self.adapt_blocks:
            num_blocks = self._nearest_divisor(n)
        else:
            num_blocks = min(self.num_blocks, n)

        block_size = n // num_blocks
        x_c = torch.zeros_like(x)

        for b in range(num_blocks):
            start = b * block_size
            end = (b + 1) * block_size if b < num_blocks - 1 else n
            x_c[start:end, start:end] = x[start:end, start:end]

        return x_c



class FisherLoader:

    def __init__(self, fisher_cache, dataset_name, device,
                 precision='fp32', compressor=None):
        self.dataset_name = dataset_name
        self.device = device
        self.fisher_cache = fisher_cache
        self.precision = precision
        self.postprocessing = None

        if compressor is None:
            self.compressor = StubCompressor()
        elif compressor == 'none':
            self.compressor = StubCompressor()
        elif compressor == 'half':
            self.compressor = HalfCompressor()
        elif compressor == 'svd':
            self.compressor = SVDCompressor()
        elif compressor == 'svd++':
            self.compressor = SVDCompressor(rank=0.25)
        elif compressor == 'svd++++':
            self.compressor = SVDCompressor(rank=0.15)
        elif compressor == 'diag':
            self.compressor = DiagonalCompressor()
        elif compressor == 'dyn8':
            self.compressor = DynamicMatrixCompressor(8)
        elif compressor == 'dyn16':
            self.compressor = DynamicMatrixCompressor(16)
        elif compressor == 'pruning':
            self.compressor = ThresholdCompressor(0.3)
        elif compressor == 'pruning++':
            self.compressor = ThresholdCompressor(0.10)
        elif compressor == 'block':
            self.compressor = BlockDiagonalCompressor()
        else:
            raise ValueError

    def load_kfac(self, task_id, only_counts=False) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        fisher_cache_path_num_aaT = fisher_cache_path.replace(".pt", "_num_aaT.pt")
        fisher_cache_path_num_ggT = fisher_cache_path.replace(".pt", "_num_ggT.pt")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if os.path.exists(fisher_cache_path_num_aaT):
                assert os.path.exists(fisher_cache_path_num_ggT)
                cur_num_aaT: int = torch.load(fisher_cache_path_num_aaT,
                                              map_location="cpu").item()
                cur_num_ggT: int = torch.load(fisher_cache_path_num_ggT,
                                              map_location="cpu").item()
            else:
                raise FileNotFoundError(
                    f"Fisher cache file {fisher_cache_path_num_aaT} or {fisher_cache_path_num_ggT} not found. ")

        if only_counts:
            return cur_num_ggT, cur_num_aaT  # type: ignore

        fisher_cache_path_aaT = fisher_cache_path.replace(".pt", "_aaT.pt")
        fisher_cache_path_ggT = fisher_cache_path.replace(".pt", "_ggT.pt")
        fisher_cache_path_ffT = fisher_cache_path.replace(".pt", "_ffT.pt")

        assert os.path.exists(fisher_cache_path_aaT)
        assert os.path.exists(fisher_cache_path_ggT)
        assert os.path.exists(fisher_cache_path_ffT)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aaT: dict = torch.load(fisher_cache_path_aaT, map_location=self.device)
            ggT: dict = torch.load(fisher_cache_path_ggT, map_location=self.device)
            ffT: dict = torch.load(fisher_cache_path_ffT, map_location=self.device)


        for key in aaT.keys():
            if self.precision == 'fp64':
                aaT[key] = self.compressor(aaT[key]).to(torch.float64)
                ggT[key] = self.compressor(ggT[key]).to(torch.float64)
            elif self.precision == 'fp32':
                aaT[key] = self.compressor(aaT[key]).to(torch.float32)
                ggT[key] = self.compressor(ggT[key]).to(torch.float32)
            else:
                raise NotImplementedError

        for key in ffT.keys():
            if self.precision == 'fp64':
                #ffT[key] = ffT[key].to(torch.float64)
                ffT[key] = self.compressor(ffT[key]).to(torch.float64)
            elif self.precision == 'fp32':
                #ffT[key] = ffT[key].to(torch.float32)
                ffT[key] = self.compressor(ffT[key]).to(torch.float32)
            else:
                raise NotImplementedError

        return ggT, aaT, ffT, cur_num_ggT, cur_num_aaT

    def store_kfac(self, task_id, ggT, aaT, ffT, num_ggT, num_aaT):
        os.makedirs(self.fisher_cache, exist_ok=True)
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        torch.save(ggT, fisher_cache_path.replace(".pt", "_ggT.pt"))
        torch.save(aaT, fisher_cache_path.replace(".pt", "_aaT.pt"))
        torch.save(ffT, fisher_cache_path.replace(".pt", "_ffT.pt"))
        torch.save(torch.tensor([num_ggT]), fisher_cache_path.replace(".pt", "_num_ggT.pt"))
        torch.save(torch.tensor([num_aaT]), fisher_cache_path.replace(".pt", "_num_aaT.pt"))

    def load_ekfac(self, task_id, only_counts=False) \
            -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:

        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        fisher_cache_path_num_of_examples = fisher_cache_path.replace(".pt", "_num_of_examples.pt")

        assert os.path.exists(fisher_cache_path_num_of_examples), f"File {fisher_cache_path_num_of_examples} not found."

        num_of_examples = \
            torch.load(fisher_cache_path_num_of_examples, map_location="cpu").item()

        if only_counts:
            return num_of_examples  # type: ignore

        fisher_cache_path_UA = fisher_cache_path.replace(".pt", "_UA.pt")
        fisher_cache_path_UG = fisher_cache_path.replace(".pt", "_UG.pt")
        fisher_cache_path_D = fisher_cache_path.replace(".pt", "_D.pt")
        fisher_cache_path_ffT = fisher_cache_path.replace(".pt", "_ffT.pt")

        assert os.path.exists(fisher_cache_path_UA)
        assert os.path.exists(fisher_cache_path_UG)
        assert os.path.exists(fisher_cache_path_D)
        assert os.path.exists(fisher_cache_path_ffT)

        UA = torch.load(fisher_cache_path_UA, map_location=self.device)
        UG = torch.load(fisher_cache_path_UG, map_location=self.device)
        D = torch.load(fisher_cache_path_D, map_location=self.device)
        ffT = torch.load(fisher_cache_path_ffT, map_location=self.device)

        assert UA.keys() == UG.keys() == D.keys()

        for key in UA.keys():
            if self.precision == 'fp64':
                UA[key] = UA[key].to(torch.float64)
                UG[key] = UG[key].to(torch.float64)
                D[key] = D[key].to(torch.float64)
            elif self.precision == 'fp32':
                UA[key] = UA[key].to(torch.float32)
                UG[key] = UG[key].to(torch.float32)
                D[key] = D[key].to(torch.float32)
            else:
                raise NotImplementedError

        for key in ffT.keys():
            if self.precision == 'fp64':
                ffT[key] = ffT[key].to(torch.float64)
            elif self.precision == 'fp32':
                ffT[key] = ffT[key].to(torch.float32)
            else:
                raise NotImplementedError

        return UA, UG, D, ffT, num_of_examples
    
    def load_diff_ekfac(self, task_id, only_counts=False) \
            -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:

        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        fisher_cache_path_num_of_examples = fisher_cache_path.replace(".pt", "_num_of_examples.pt")

        assert os.path.exists(fisher_cache_path_num_of_examples)

        num_of_examples = \
            torch.load(fisher_cache_path_num_of_examples, map_location="cpu").item()

        if only_counts:
            return num_of_examples  # type: ignore

        fisher_cache_universe_path = f"{self.fisher_cache}/{self.dataset_name}_universe.pt"
        fisher_cache_path_UA = fisher_cache_universe_path.replace(".pt", "_UA.pt")
        fisher_cache_path_UG = fisher_cache_universe_path.replace(".pt", "_UG.pt")
        
        fisher_cache_path_D = fisher_cache_path.replace(".pt", "_D.pt")
        fisher_cache_path_ffT = fisher_cache_path.replace(".pt", "_ffT.pt")

        assert os.path.exists(fisher_cache_path_UA)
        assert os.path.exists(fisher_cache_path_UG)
        assert os.path.exists(fisher_cache_path_D)
        assert os.path.exists(fisher_cache_path_ffT)
        if task_id == 0:
            UA = torch.load(fisher_cache_path_UA, map_location=self.device)
            UG = torch.load(fisher_cache_path_UG, map_location=self.device)
        else:
            UA = {}
            UG = {}
        D = torch.load(fisher_cache_path_D, map_location=self.device)
        ffT = torch.load(fisher_cache_path_ffT, map_location=self.device)

        if task_id == 0:
            assert UA.keys() == UG.keys() == D.keys()

        for key in UA.keys():
            if self.precision == 'fp64':
                UA[key] = UA[key].to(torch.float64)
                UG[key] = UG[key].to(torch.float64)
                D[key] = D[key].to(torch.float64)
            elif self.precision == 'fp32':
                UA[key] = UA[key].to(torch.float32)
                UG[key] = UG[key].to(torch.float32)
                D[key] = D[key].to(torch.float32)
            else:
                raise NotImplementedError

        for key in ffT.keys():
            if self.precision == 'fp64':
                ffT[key] = ffT[key].to(torch.float64)
            elif self.precision == 'fp32':
                ffT[key] = ffT[key].to(torch.float32)
            else:
                raise NotImplementedError

        return UA, UG, D, ffT, num_of_examples

    def load_diag(self, task_id, only_counts=False) \
            -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:

        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        fisher_cache_path_num_of_examples = fisher_cache_path.replace(".pt", "_num_of_examples.pt")

        assert os.path.exists(fisher_cache_path_num_of_examples)

        num_of_examples = \
            torch.load(fisher_cache_path_num_of_examples, map_location="cpu").item()

        if only_counts:
            return num_of_examples # type: ignore

        fisher_cache_path_ffT = fisher_cache_path.replace(".pt", "_ffT.pt")

        assert os.path.exists(fisher_cache_path_ffT)

        ffT = torch.load(fisher_cache_path_ffT, map_location=self.device)

        for key in ffT.keys():
            if self.precision == 'fp64':
                ffT[key] = ffT[key].to(torch.float64)
            elif self.precision == 'fp32':
                ffT[key] = ffT[key].to(torch.float32)
            else:
                raise NotImplementedError

        return ffT, num_of_examples

    def store_ekfac(self, task_id, UA, UG, D, ffT, num_of_examples):
        os.makedirs(self.fisher_cache, exist_ok=True)
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        torch.save(UA, fisher_cache_path.replace(".pt", "_UA.pt"))
        torch.save(UG, fisher_cache_path.replace(".pt", "_UG.pt"))
        torch.save(D, fisher_cache_path.replace(".pt", "_D.pt"))
        torch.save(ffT, fisher_cache_path.replace(".pt", "_ffT.pt"))
        torch.save(torch.tensor([num_of_examples]), fisher_cache_path.replace(".pt", "_num_of_examples.pt"))
        
    def store_diff_ekfac(self, task_id, UA, UG, D, ffT, num_of_examples):
        os.makedirs(self.fisher_cache, exist_ok=True)
        if task_id == 0:
            fisher_cache_universe_path = f"{self.fisher_cache}/{self.dataset_name}_universe.pt"
            torch.save(UA, fisher_cache_universe_path.replace(".pt", "_UA.pt"))
            torch.save(UG, fisher_cache_universe_path.replace(".pt", "_UG.pt"))
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        torch.save(D, fisher_cache_path.replace(".pt", "_D.pt"))
        torch.save(ffT, fisher_cache_path.replace(".pt", "_ffT.pt"))
        torch.save(torch.tensor([num_of_examples]), fisher_cache_path.replace(".pt", "_num_of_examples.pt"))


    def store_diag(self, task_id, ffT, num_of_examples):
        os.makedirs(self.fisher_cache, exist_ok=True)
        fisher_cache_path = f"{self.fisher_cache}/{self.dataset_name}_task_{task_id}.pt"
        torch.save(ffT, fisher_cache_path.replace(".pt", "_ffT.pt"))
        torch.save(torch.tensor([num_of_examples]), fisher_cache_path.replace(".pt", "_num_of_examples.pt"))


def get_parameter(shape, device, type_init: str = 'orto',
                  transpose: bool = False, requires_grad: bool = True):
    param = torch.zeros(*shape, dtype=torch.float32, device=device)
    if type_init == 'orto':
        torch.nn.init.orthogonal_(param)
    if type_init == 'gaussian':
        torch.nn.init.normal_(param, mean=0.0, std=0.1)
    if type_init == 'kernel':
        torch.nn.init.normal_(param, mean=0.0, std=0.036)
    if type_init == 'attn':
        torch.nn.init.normal_(param, mean=1.0, std=0.03)
    if type_init == 'kaiming':
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    if type_init == 'ones':
        torch.nn.init.ones_(param)
    if transpose:
        param = torch.transpose(param, 1, 2)
    return torch.nn.Parameter(param, requires_grad=requires_grad)


def get_params(net, features=True, classifier=False, offset_1=-1, offset_2=-1) -> torch.Tensor:
    params = []
    for name, param in net.named_parameters():
        if "head" in name:
            if classifier:
                assert (offset_1 > -1 and offset_2 > -1)
                params.append(param[offset_1:offset_2].view(-1))
        elif features:
            params.append(param.view(-1))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.])

def set_params(net, new_params: torch.Tensor, features=True,
               classifier=False, offset_1=-1, offset_2=-1) -> None:
    progress = 0
    for name, param in net.named_parameters():
        if "head" in name:
            if classifier:
                assert (offset_1 > -1 and offset_2 > -1)
                cur_size = torch.tensor(param.data[offset_1:offset_2].size()).prod()
                param.data[offset_1:offset_2] = new_params[progress: progress + cur_size].view(
                    param.data[offset_1:offset_2].size())
                progress += cur_size
        elif features:
            cur_size = torch.tensor(param.size()).prod()
            cand_params = new_params[progress: progress + cur_size].view(param.size())
            param.data = cand_params
            progress += cur_size

def get_delta_w_backbone(named_params, delta_w, delta_w_names, training_type, device):
    params = []
    for name, param in named_params():
        name = name.replace("visual_encoder.", "")
        if "head" not in name:
            if name in delta_w_names:
                index = delta_w_names.index(name)
                cur_delta_w = delta_w[index]
                params.append(cur_delta_w.view(-1).to(device))
            elif name == "logit_scale":
            #else:
                #params.append(torch.zeros_like(param).view(-1).to(device))
                print(name)
                print("ops siamo finiti in sto posto strano ma non facciamo nulla")
                #params.append(torch.clone(param).view(-1).to(device))

    if len(params):
        return torch.cat(params)
    else:
        return torch.tensor([0.]).to(device)

def get_delta_w_parameterlist(named_params, delta_w, delta_w_names, peft_type, device):
    params = []
    for name, param in named_params():
        if name in delta_w_names:
            index = delta_w_names.index(name)
            if peft_type == "lora":
                cur_delta_w = delta_w[index][0] @ delta_w[index][1]
            elif peft_type == "full":
                cur_delta_w = delta_w[index]
            params.append(cur_delta_w.to(device))
        else:
            params.append(torch.zeros_like(param).to(device))

    return params

def replace_non_dynamically_quantizable_linear(module):
    """Recursively replace all NonDynamicallyQuantizableLinear layers with Linear layers in a model."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
            # Replace with an equivalent Linear layer
            new_layer = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
            new_layer.weight = torch.nn.Parameter(child.weight.clone())  # Copy weights
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.clone())  # Copy bias
            setattr(module, name, new_layer)
        else:
            replace_non_dynamically_quantizable_linear(child)  # Recursively process children

    return module

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
        for param_group, base_lr, group_min_lr in zip(optimizer.param_groups, base_lrs, min_lr_list):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
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
                progress = step / steps
                if progress < 0.70:
                    lr = base_lr
                elif progress < 0.90:
                    lr = base_lr * 0.5
                else:
                    lr = base_lr * 0.1
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

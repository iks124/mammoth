from contextlib import suppress
from copy import deepcopy
import gc
import json
import os
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.func as func
import torch.nn as nn

from datasets.utils.continual_dataset import ContinualDataset
from models.clip_ft_utils.backbone import Backbone
from models.clip_ft_utils.backbone import build_classification_head
from models.clip_ft_utils.backbone import create_clip
from models.clip_ft_utils.eigenvals import compute_eigenvalues
from models.clip_ft_utils.fisher_kfac import KFACComputer
from models.clip_ft_utils.merging import add_merging_args
from models.clip_ft_utils.merging import get_merging_function
from models.clip_ft_utils.utils import FisherLoader
from models.clip_ft_utils.utils import OptimizerBuilder
from models.clip_ft_utils.utils import add_clip_args
from models.clip_ft_utils.utils import compute_acc_on_last_task
from models.clip_ft_utils.utils import get_parameter
from models.utils.continual_model import ContinualModel
from utils import binary_to_boolean_type
from utils.args import ArgumentParser
from utils.conf import get_device
from utils.evaluate import evaluate as evaluate_all_tasks

with suppress(ImportError):
    import wandb


class TAK(ContinualModel):
    NAME = "tak"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]
    net: Backbone

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_clip_args(parser)
        add_merging_args(parser)

        parser.add_argument(
            "--save_task_vectors", type=binary_to_boolean_type, default=0
        )
        parser.add_argument("--tangent", type=binary_to_boolean_type, default=1)
        parser.add_argument("--chunks", type=int, default=1)
        parser.add_argument("--use_lora", type=binary_to_boolean_type, default=0)
        parser.add_argument("--load_fisher", type=binary_to_boolean_type, default=0)
        parser.add_argument("--layer_scale_lambda", type=float, default=1)
        parser.add_argument(
            "--default_scale_factor", type=float, default=1, choices=[0, 1]
        )
        parser.add_argument("--fisher_num_samples_expectation", type=int, default=0)
        parser.add_argument("--fisher_use_head", type=binary_to_boolean_type, default=0)
        parser.add_argument(
            "--fisher_use_softmax", type=binary_to_boolean_type, default=0
        )
        parser.add_argument("--fisher_ideal", type=binary_to_boolean_type, default=0)
        parser.add_argument("--fisher_norm_scaler", type=float, default=1e4)
        parser.add_argument("--fisher_ft_proj_scaler", type=float, default=1e-3)
        parser.add_argument("--reg_lambda", type=float, default=0)
        parser.add_argument(
            "--scheduler_ntk",
            type=str,
            default="cosine_plus",
            choices=["none", "cosine", "cosine_plus", "decay", "step"],
        )
        parser.add_argument("--fisher_cache", type=str, default="fisher_cache")
        parser.add_argument("--train_percent", type=str, default="1.0")
        parser.add_argument("--fisher_task_id", type=int, default=None, required=False)
        parser.add_argument(
            "--clip_grad_norm", type=float, default=None, required=False
        )
        parser.add_argument(
            "--precision", type=str, default="fp32", choices=["fp32", "fp64"]
        )
        parser.add_argument("--load_path", type=str, default=None, required=False)
        parser.add_argument("--resume", type=binary_to_boolean_type, default=0)
        parser.add_argument("--kfac_rate", type=int, default=1, required=False)
        parser.add_argument(
            "--kfac_quantization",
            type=str,
            default="none",
            choices=[
                "none",
                "svd",
                "diag",
                "dyn8",
                "dyn16",
                "pruning",
                "block",
                "svd++",
                "svd++++",
                "pruning++",
            ],
        )
        parser.add_argument("--sweep_start", type=float, default=0.1)
        parser.add_argument("--sweep_end", type=float, default=1.5)
        parser.add_argument("--sweep_step", type=float, default=0.1)
        parser.add_argument("--save_path", type=str, default="checkpoints")
        return parser

    def parse_train_percent(self):
        if not isinstance(self.args.train_percent, str):
            return self.args
        raw = self.args.train_percent.strip()
        if ("." in raw) or ("e" in raw) or ("E" in raw):
            val = float(raw)
            assert 0 < val <= 1
            self.args.train_percent = val
        else:
            val = int(raw)
            assert val >= 1
            self.args.train_percent = val
        return self.args

    def __init__(self, backbone, loss, args, transform, dataset):
        assert dataset is not None
        assert 0 <= args.layer_scale_lambda <= 1

        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "Please install CLIP: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        _, clip_preprocess = clip.load(
            args.clip_backbone, device=torch.device("cpu"), jit=False
        )
        clip_model = create_clip(args.clip_backbone, get_device())

        super().__init__(clip_model, loss, args, transform, dataset=dataset)  # type: ignore

        self.parse_train_percent()
        if self.args.resume:
            assert self.args.load_path is not None

        if self.args.save_task_vectors:
            task_vector_path = (
                f"checkpoints/{self.args.fisher_cache}"
                if self.args.save_path == "checkpoints"
                else self.args.save_path
            )
            os.makedirs(task_vector_path, exist_ok=True)
            with open(
                f"{task_vector_path}/{self.args.conf_jobnum}_{dataset.NAME}_args.json",
                "w",
                encoding="utf-8",
            ) as f:
                json_args = deepcopy(vars(args))
                if "device" in json_args:
                    del json_args["device"]
                json.dump(json_args, f)

        self.net = Backbone(clip_model, dataset, args)
        self.param_names = [
            name for name, _ in self.net.visual_encoder.named_parameters()
        ]

        for _, param in self.net.named_parameters():
            param.requires_grad = False

        with suppress(Exception):
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        clip_model = clip_model.to(dtype=torch.float32)
        clip_model.eval()
        self.clip_model = clip_model
        self.clip_transform = clip_preprocess
        self.clip_eval_transform = clip_preprocess

        self.fisher_computer = KFACComputer(
            self.device,
            self.args.debug_mode == 1,
            train_percent=self.args.train_percent,
            num_samples_expectation=self.args.fisher_num_samples_expectation,
            precision=self.args.precision,
        )
        self.fisher_loader = FisherLoader(
            self.args.fisher_cache,
            dataset.NAME,
            self.device,
            precision="fp32",
            compressor=self.args.kfac_quantization,
        )
        self.optimizer_builder = OptimizerBuilder(cmd_args=self.args)

        self.cur_offset = None
        self.cls_head: nn.Module = None  # type: ignore
        self.delta_w_dict: dict[str, Any] = None  # type: ignore
        self.delta_w_names: list[str] = None  # type: ignore
        self.delta_w_shapes: dict[str, Any] = None  # type: ignore
        self.scheduler1 = None
        self.tasks_ggT = {}
        self.tasks_ffT = {}
        self.tasks_aaT = {}
        self.coeffs = []
        self.layer_scale_factors: dict[str, float] = {}
        self.merging = get_merging_function(self.args, self.device)
        self.merged_task_vector = []
        self.num_total_tasks: int = dataset.N_TASKS  # type: ignore
        self.dataset_name: str = dataset.NAME  # type: ignore
        self.individual_acc, self.individual_mask_acc = [], []
        self.norm_acc, self.norm_mask_acc = [], []
        self.task_loaded = False

    def create_param_like(self, param, requires_grad):
        return [
            torch.zeros_like(
                param,
                dtype=torch.float32,
                requires_grad=requires_grad,
                device=self.args.device,
            )
        ]

    def create_lora_param_like(self, fin, fout, requires_grad, r1=None, r2=None):
        r1 = 16 if r1 is None else r1
        r2 = 16 if r2 is None else r2
        return (
            get_parameter((fout, r2), self.device, "zeros", False, requires_grad),
            get_parameter((r1, fin), self.device, "kaiming", False, requires_grad),
        )

    def begin_task(self, dataset):
        torch.cuda.empty_cache()
        dataset.test_loaders[-1].dataset.transform = self.clip_eval_transform
        dataset.train_loader.dataset.transform = self.clip_transform  # type: ignore
        self.cur_offset = self.get_offsets(self.current_task)

        if isinstance(dataset.N_CLASSES_PER_TASK, int):
            self.cpt = dataset.N_CLASSES_PER_TASK
        else:
            self.cpt = dataset.N_CLASSES_PER_TASK[-1]

        if self.current_task != 0:
            self.net.task_id += 1

        self.cls_head = build_classification_head(
            self.clip_model, dataset, self.cur_offset
        )
        self.net.copy_visual_encoder(self.clip_model)

        for param in self.net.visual_encoder.parameters():
            param.requires_grad = False

        self.delta_w_dict = {}
        self.delta_w_shapes = {}
        for name, param in self.net.visual_encoder.named_parameters():
            self.delta_w_shapes[name] = param.shape
            if self.args.use_lora == 1 and len(param.shape) == 2:
                fout, fin = param.shape[0], param.shape[1]
                if "mlp" in name:
                    b, a = self.create_lora_param_like(
                        fin, fout, self.args.ft_linears == 1
                    )
                    self.delta_w_dict[name] = [b, a]
                elif "attn" in name:
                    b, a = self.create_lora_param_like(
                        fin, fout, self.args.ft_attention == 1, r1=16 * 3, r2=16 * 3
                    )
                    self.delta_w_dict[name] = [b, a]
                elif "proj" in name:
                    b, a = self.create_lora_param_like(
                        fin, fout, self.args.ft_proj == 1
                    )
                    self.delta_w_dict[name] = [b, a]
            else:
                if "mlp" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_linears == 1
                    )
                elif "attn" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_attention == 1
                    )
                elif "proj" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_proj == 1
                    )
                elif "ln" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_ln == 1
                    )
                elif "class_embed" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_class_embed == 1
                    )
                elif "conv" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_conv == 1
                    )
                elif "positional_embedding" in name:
                    self.delta_w_dict[name] = self.create_param_like(
                        param, requires_grad=self.args.ft_pos_embed == 1
                    )

        self.delta_w_names = list(self.delta_w_dict.keys())

        if not self.args.load_fisher:
            if (
                self.args.fisher_task_id is None
                or self.current_task == self.args.fisher_task_id
            ):
                dataset.train_loader.dataset.transform = self.clip_eval_transform  # type: ignore
                head = None
                assert (self.args.fisher_use_softmax + self.args.fisher_use_head) <= 1
                if self.args.fisher_use_head:
                    head = build_classification_head(
                        self.clip_model,
                        dataset,
                        self.get_offsets(self.num_total_tasks - 1),
                        eval=True,
                    )
                elif self.args.fisher_use_softmax:
                    head = build_classification_head(
                        self.clip_model,
                        dataset,
                        self.get_offsets(self.num_total_tasks - 1),
                        eval=True,
                    )
                    head = nn.Sequential(head, nn.Softmax(dim=1))
                gg_t, aa_t, ff_t, num_gg_t, num_aa_t = self.fisher_computer.compute(
                    self.net,
                    head,
                    self.delta_w_names,
                    dataset,
                    use_head=head is not None,
                )
                dataset.train_loader.dataset.transform = self.clip_transform  # type: ignore
                self.fisher_loader.store_kfac(
                    self.current_task, gg_t, aa_t, ff_t, num_gg_t, num_aa_t
                )
        else:
            counts = [
                self.fisher_loader.load_kfac(t, only_counts=True)
                for t in range(dataset.N_TASKS)
            ]  # type: ignore
            tot_gg_t = sum(
                [
                    cnt[0]
                    for idx_cnt, cnt in enumerate(counts)
                    if idx_cnt != self.current_task
                ]
            )
            tot_aa_t = sum(
                [
                    cnt[1]
                    for idx_cnt, cnt in enumerate(counts)
                    if idx_cnt != self.current_task
                ]
            )
            assert tot_gg_t == tot_aa_t
            self.coeffs = []
            num_penalties = dataset.N_TASKS - 1 if self.args.fisher_ideal else 1
            for t in range(dataset.N_TASKS):
                gg_t, aa_t, ff_t, cur_num_gg_t, cur_num_aa_t = (
                    self.fisher_loader.load_kfac(t)
                )
                assert cur_num_gg_t == cur_num_aa_t
                coeff = cur_num_gg_t / tot_gg_t
                if t == 0:
                    for key in aa_t.keys():
                        if key in self.tasks_ggT.keys():
                            for p_l in range(num_penalties):
                                self.tasks_aaT[key][p_l].zero_()
                                self.tasks_ggT[key][p_l].zero_()
                        else:
                            self.tasks_aaT[key] = [
                                torch.zeros_like(aa_t[key])
                                for _ in range(num_penalties)
                            ]
                            self.tasks_ggT[key] = [
                                torch.zeros_like(gg_t[key])
                                for _ in range(num_penalties)
                            ]
                    for key in ff_t.keys():
                        if key in self.tasks_ffT.keys():
                            self.tasks_ffT[key].zero_()
                        else:
                            self.tasks_ffT[key] = torch.zeros_like(ff_t[key])
                if t != self.current_task:
                    self.coeffs.append(coeff)
                    for key in ff_t.keys():
                        self.tasks_ffT[key].add_(ff_t[key] / tot_gg_t)
                    for key in aa_t.keys():
                        aa_t[key].div_(cur_num_aa_t)
                        gg_t[key].div_(cur_num_gg_t)
                        if self.args.fisher_ideal == 0:
                            self.tasks_aaT[key][0].add_(
                                (cur_num_aa_t / tot_aa_t) * aa_t[key]
                            )
                            self.tasks_ggT[key][0].add_(gg_t[key])
                        else:
                            t_hat = t if t <= self.current_task else t - 1
                            self.tasks_ggT[key][t_hat].copy_(gg_t[key])
                            self.tasks_aaT[key][t_hat].copy_(aa_t[key])

        if self.args.layer_scale_lambda < 1:
            assert self.args.fisher_ideal == 0
            with torch.no_grad():
                layer_eigens = compute_eigenvalues(
                    self.tasks_ggT, self.tasks_aaT, self.tasks_ffT
                )
                layer_groups = [
                    "attn.qkv",
                    "attn.proj",
                    "mlp",
                    "ln_",
                    "lin_proj",
                    "class_embedding",
                ]
                for layer_group in layer_groups:
                    for wb in ["weight", "bias", "class_embedding"]:
                        eighs = [
                            eigens
                            for lname, eigens in layer_eigens.items()
                            if layer_group in lname and wb in lname
                        ]
                        if len(eighs) > 0:
                            neigs = len([eigens.max().item() for eigens in eighs])
                            max_eigh = max([eigens.max().item() for eigens in eighs])
                            for lname, eigens in layer_eigens.items():
                                if layer_group in lname and wb in lname:
                                    c_eigh = eigens.max().item()
                                    scale_factor = (
                                        self.args.default_scale_factor
                                        if neigs == 1
                                        else (max_eigh - c_eigh) / max_eigh
                                    )
                                    self.layer_scale_factors[lname] = scale_factor

        all_params = [
            p for param_list in self.delta_w_dict.values() for p in param_list
        ]
        num_batches = len(dataset.train_loader)  # type: ignore
        self.opt, self.scheduler1 = self.optimizer_builder.build_opt_and_sched(
            all_params, num_batches
        )

        if self.args.resume:
            self.task_loaded = self.load_task_vectors()
            if self.task_loaded:
                self.args.n_epochs = 0
                self.n_epochs = 0

        self.train()

    def get_parameter_from_dict(self, name):
        assert name in self.delta_w_names
        list_params = self.delta_w_dict[name]
        if len(list_params) == 1:
            return list_params[0]
        if len(list_params) == 2:
            return list_params[0] @ list_params[1]
        raise ValueError

    def get_all_parameters_from_dict(self):
        return [self.get_parameter_from_dict(k) for k in self.delta_w_names]

    def end_task(self, dataset: ContinualDataset) -> None:
        self.eval()
        self.merged_task_vector = [
            torch.clone(self.get_parameter_from_dict(key)) for key in self.delta_w_names
        ]
        actual_seen_classes = self.n_seen_classes
        self.cls_head = build_classification_head(
            self.clip_model, dataset, self.cur_offset, all_heads=True
        )
        self._n_seen_classes = dataset.N_CLASSES
        acc, acc_mask_classes = compute_acc_on_last_task(self, dataset)
        self.individual_acc.append(acc)
        self.individual_mask_acc.append(acc_mask_classes)
        self._n_seen_classes = actual_seen_classes

        if self.args.save_task_vectors:
            save_data = [
                self.get_parameter_from_dict(key).clone().cpu()
                for key in self.delta_w_names
            ]
            base_name = (
                f"{self.args.conf_jobnum}_{dataset.NAME}_task_{self.current_task}"
            )
            if self.args.save_path == "checkpoints":
                base_path = f"checkpoints/{self.args.fisher_cache}/{base_name}"
            else:
                base_path = os.path.join(self.args.save_path, base_name)
            tv_path = base_path + ".pt"
            os.makedirs(os.path.dirname(tv_path), exist_ok=True)
            torch.save(save_data, tv_path)
            torch.save(self.cls_head.state_dict(), base_path + "_cls_head.pt")
            torch.save({"delta_w_names": self.delta_w_names}, base_path + "_meta.pt")

        self.cls_head = build_classification_head(
            self.clip_model, dataset, self.cur_offset, eval=True
        )
        for key in self.delta_w_names:
            for p in self.delta_w_dict[key]:
                p.requires_grad = False

        self.merging.add(
            {key: self.get_parameter_from_dict(key) for key in self.delta_w_names}
        )
        self.merged_task_vector = self.merging.merge(self.delta_w_names)

        if self.args.layer_scale_lambda < 1:
            for i, key in enumerate(self.delta_w_names):
                if key in self.layer_scale_factors:
                    eigh_scale = self.layer_scale_factors[key]
                    scale_factor = (
                        self.args.layer_scale_lambda
                        + (1 - self.args.layer_scale_lambda) * eigh_scale
                    )
                    self.merged_task_vector[i].mul_(scale_factor)

        if self.opt is not None:
            self.opt.zero_grad()  # type: ignore
        self.opt = None
        self.net.copy_visual_encoder(self.clip_model)
        torch.cuda.empty_cache()
        del self.scheduler1, self.delta_w_dict
        gc.collect()
        return super().end_task(dataset)

    def end_eval(self, dataset: ContinualDataset, accs: Tuple[List, List]) -> None:
        def safe_den(y, eps=1e-8):
            return y if abs(y) >= eps else y + eps

        self.norm_acc = [
            acc / safe_den(self.individual_acc[t]) for t, acc in enumerate(accs[0])
        ]
        self.norm_mask_acc = [
            acc / safe_den(self.individual_mask_acc[t]) for t, acc in enumerate(accs[1])
        ]

        if self.args.nowand == 0 and "wandb" in globals():
            wandb.log(
                {
                    "RESULT_mean_norm_acc": sum(self.norm_acc) / len(self.norm_acc),
                    "RESULT_mean_norm_mask_acc": sum(self.norm_mask_acc)
                    / len(self.norm_mask_acc),
                    "Task": self.current_task,
                }
            )
        if self.current_task == self.num_total_tasks:
            self.alpha_sweep()

    def penalty_weight(self):
        loss_reg, loss_reg_ff_t, loss_ft_proj, loss_reg_cls_emb = 0, 0, 0, 0
        for name in self.delta_w_names:
            if name in self.tasks_aaT.keys():
                delta_w = self.get_parameter_from_dict(name)
                bias_name = name.replace("weight", "bias")
                if bias_name in self.delta_w_names:
                    delta_bias = self.get_parameter_from_dict(bias_name)
                    delta_w = torch.cat((delta_w, delta_bias.unsqueeze(1)), 1)
                for task_id in range(len(self.tasks_aaT[name])):
                    aa_t_past = self.tasks_aaT[name][task_id]
                    gg_t_past = self.tasks_ggT[name][task_id]
                    norm_coeff = self.coeffs[task_id] if self.args.fisher_ideal else 1
                    loss_ = torch.trace(gg_t_past @ delta_w @ aa_t_past @ delta_w.T)
                    if name == "lin_proj.weight":
                        loss_ft_proj += norm_coeff * loss_
                    else:
                        loss_reg += norm_coeff * loss_
            if name in self.tasks_ffT.keys():
                delta_w = self.get_parameter_from_dict(name).unsqueeze(0)
                ff_t_past = self.tasks_ffT[name]
                reg_w = torch.trace(delta_w @ ff_t_past @ delta_w.T)
                if "class_embedding" in name:
                    loss_reg_cls_emb += reg_w
                else:
                    loss_reg_ff_t += reg_w
        return loss_reg, loss_reg_ff_t, loss_ft_proj, loss_reg_cls_emb

    def create_functional(self, inputs, delta_names):
        def func_network(param_values):
            param = {name: param for name, param in zip(delta_names, param_values)}
            features = func.functional_call(self.net.visual_encoder, param, inputs)  # type: ignore
            return nn.functional.normalize(features, dim=-1)

        return func_network

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.args.resume and self.task_loaded:
            return 0.0

        if self.args.tangent:
            forward_fun = self.create_functional(inputs, self.delta_w_names)
            params = [
                param
                for name, param in self.net.visual_encoder.named_parameters()
                if name in self.delta_w_names
            ]
            image_features, jvp = func.jvp(
                forward_fun,
                (tuple(params),),
                (tuple(self.get_all_parameters_from_dict()),),
            )
            image_features = image_features + jvp
        else:
            tunable_params = [
                p
                for n, p in self.net.visual_encoder.named_parameters()
                if n in self.delta_w_names
            ]
            dict_param = {
                name: param + net_param
                for name, param, net_param in zip(
                    self.delta_w_names,
                    self.get_all_parameters_from_dict(),
                    tunable_params,
                )
            }
            image_features = func.functional_call(
                self.net.visual_encoder, dict_param, inputs
            )  # type: ignore
            image_features = nn.functional.normalize(image_features, dim=-1)

        similarity = self.cls_head(image_features)
        loss_task = self.loss(similarity, labels - self.n_past_classes)
        loss = loss_task / self.args.chunks
        loss.backward()

        chunk_id = self.task_iteration // self.args.chunks
        if (
            self.args.load_fisher
            and self.task_iteration > 0
            and self.task_iteration % self.args.chunks == 0
            and chunk_id % self.args.kfac_rate == 0
        ):
            loss_penalty, loss_reg_ff_t, loss_ft_proj, loss_reg_cls_emb = (
                self.penalty_weight()
            )
            loss_reg = (
                self.args.reg_lambda * loss_penalty
                + self.args.reg_lambda * self.args.fisher_norm_scaler * loss_reg_ff_t
                + self.args.reg_lambda * self.args.fisher_ft_proj_scaler * loss_ft_proj
                + self.args.reg_lambda * self.args.fisher_norm_scaler * loss_reg_cls_emb
            )
            loss_reg.backward()

        if self.task_iteration > 0 and self.task_iteration % self.args.chunks == 0:
            if self.scheduler1:
                self.scheduler1(self.task_iteration // self.args.chunks)
            if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for group in self.opt.param_groups for p in group["params"]),
                    self.args.clip_grad_norm,
                )
            self.opt.step()  # type: ignore
            self.opt.zero_grad()  # type: ignore

        return loss.item()

    @torch.no_grad()
    def forward(self, x):
        if self.args.tangent:
            forward_fun = self.create_functional(x, self.delta_w_names)
            params = [
                param
                for name, param in self.net.visual_encoder.named_parameters()
                if name in self.delta_w_names
            ]
            image_features, jvp = func.jvp(
                forward_fun, (tuple(params),), (tuple(self.merged_task_vector),)
            )
            image_features = image_features + jvp
        else:
            tunable_params = {
                n: p
                for n, p in self.net.visual_encoder.named_parameters()
                if n in self.delta_w_names
            }
            dict_param = {
                key: tunable_params[key] + self.merged_task_vector[i]
                for i, key in enumerate(self.delta_w_names)
            }
            image_features = func.functional_call(
                self.net.visual_encoder, dict_param, x
            )  # type: ignore
            image_features = nn.functional.normalize(image_features, dim=-1)
        similarity = self.cls_head(image_features)
        return similarity[:, : self.n_seen_classes]

    def get_debug_iters(self):
        return 5

    def load_task_vectors(self):
        tv_path = self.args.load_path.replace(
            "_args.json", f"_task_{self.current_task}.pt"
        )
        if not os.path.exists(tv_path):
            return False
        task_vectors = torch.load(tv_path, map_location=self.device)
        meta_path = tv_path.replace(".pt", "_meta.pt")
        if not os.path.exists(meta_path):
            return False
        meta_data = torch.load(meta_path, map_location=self.device)
        delta_w_names = meta_data["delta_w_names"]
        for name, param in zip(delta_w_names, task_vectors):
            self.delta_w_dict[name] = [param.clone().detach().to(self.device)]
        return True

    def alpha_sweep(self):
        if "wandb" not in globals():
            return

        def safe_den(y, eps=1e-8):
            return y if abs(y) >= eps else y + eps

        if not hasattr(self, "_alpha_sweep_next_id"):
            self._alpha_sweep_next_id = 1
        sweep_id = self._alpha_sweep_next_id
        self._alpha_sweep_next_id += 1
        metric_name = "alpha_sweep" if sweep_id == 1 else f"alpha_sweep_{sweep_id}"
        norm_metric_name = (
            "norm_alpha_sweep" if sweep_id == 1 else f"norm_alpha_sweep_{sweep_id}"
        )

        if not hasattr(self, "_alpha_metric_defined"):
            wandb.define_metric("alpha")
            self._alpha_metric_defined = True
        wandb.define_metric(metric_name, step_metric="alpha")
        wandb.define_metric(norm_metric_name, step_metric="alpha")

        alphas = np.arange(
            self.args.sweep_start,
            self.args.sweep_end + self.args.sweep_step,
            self.args.sweep_step,
        ).tolist()
        alphas = [a * self.num_total_tasks for a in alphas]
        for alpha in alphas:
            self.merging.set_alpha(alpha)
            self.merged_task_vector = self.merging.merge(self.delta_w_names)
            _, accs_mask_classes = evaluate_all_tasks(self, self.dataset)
            norm_mask_acc = [
                acc / safe_den(self.individual_mask_acc[t])
                for t, acc in enumerate(accs_mask_classes)
            ]
            wandb.log(
                {
                    "alpha": alpha,
                    metric_name: sum(accs_mask_classes) / len(accs_mask_classes),
                    norm_metric_name: sum(norm_mask_acc) / len(norm_mask_acc),
                }
            )

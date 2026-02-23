from numpy import double
from regex import D
from sympy import numer
import torch
from torch import nn
from tqdm import tqdm
import math

from models.tak_utils.hooks import hook_forward_store_inputs
from models.tak_utils.fisher_kfac import KFACComputer

from models.tak_utils.utils import FisherLoader
from models.tak_utils.utils import set_requires_grad_to


def get_split(dataset):
    return dataset.train_loader


@torch.no_grad()
def hook_backward_ekfac(module, _, grad_output):
    if module.precision == 'fp32':
        grad_out = grad_output[0].float()
        inputs = module.inputs.float()
    elif module.precision == 'fp64':
        grad_out = grad_output[0].double()
        inputs = module.inputs.double()
    else:
        raise ValueError(f"Precision {module.precision} not supported.")

    if len(grad_out.shape) > 2:
        if 'attn.proj' in module.name or 'attn.qkv' in module.name:
            B, R, C = grad_out.shape
        else:
            R, B, C = grad_out.shape
            grad_out = grad_out.permute(1, 0, 2)
            inputs = inputs.permute(1, 0, 2)
        grad_weight = torch.einsum('blo,bli->boi', grad_out, inputs)
    else:
        grad_weight = torch.einsum('bo,bi->boi', grad_out, inputs)

    grad_bias = None

    if hasattr(module, "bias") and module.compute_bias:
        if len(grad_out.shape) > 2:
            grad_bias = grad_out.sum(1)
        else:
            assert False

    if grad_bias is not None:
        grad_weight = torch.cat((grad_weight, grad_bias.unsqueeze(2)), dim=2)

    grad_weight = torch.einsum('ij,bjk->bik', module.UG.T, grad_weight)
    grad_weight = torch.einsum('bij,jk->bik', grad_weight, module.UA)
    grad_weight = grad_weight.pow(2).sum(0)

    # --- Gram bias ---
    if not hasattr(module, "grad_weight"):
        module.grad_weight = torch.zeros_like(grad_weight)
        module.grad_weight_c = torch.zeros_like(grad_weight)

    # Kahan summation
    y_b = grad_weight - module.grad_weight_c
    t_b = module.grad_weight + y_b
    module.grad_weight_c = (t_b - module.grad_weight) - y_b
    module.grad_weight = t_b


def register_hooks(name, module, forward=True, backward=True,
                   forward_hooks_dict=None, backward_hooks_dict=None):
    module.name = name

    if forward:
        assert forward_hooks_dict is not None
        if 'lin_proj' in name:
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward_nosequence']) # type: ignore
        elif isinstance(module, nn.Linear) or \
                isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward']) # type: ignore
        elif isinstance(module, nn.LayerNorm):
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward_layer_norm']) # type: ignore
        elif 'cls_token' in name:
            module.forward_handle = module.register_forward_hook(forward_hooks_dict['hook_forward_layer_norm']) # type: ignore

    if backward:
        assert backward_hooks_dict is not None
        if 'lin_proj' in name:
            module.backward_handle = module.register_full_backward_hook(
                backward_hooks_dict['hook_backward_nosequence'])  # type: ignore
        elif isinstance(module, nn.Linear) or \
                isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            module.backward_handle = module.register_full_backward_hook(
                backward_hooks_dict['hook_backward'])  # type: ignore
        elif isinstance(module, nn.LayerNorm):
            module.backward_handle = module.register_full_backward_hook(
                backward_hooks_dict['hook_backward_layer_norm'])  # type: ignore
        elif 'cls_token' in name:
            module.backward_handle = module.register_full_backward_hook(backward_hooks_dict['hook_backward_cls_token']) # type: ignore


class EKFAComputer(nn.Module):

    def __init__(self, device: torch.device, debug_mode,
                 fisher_loader: FisherLoader = None, train_percent: float = 1.0,
                 num_samples_expectation: int = 0, precision: str = 'fp64'):

        super().__init__()

        assert 0 < train_percent <= 1.0

        self.device = device
        self.debug_mode = debug_mode
        self.train_percent = train_percent
        self.precision = precision
        
        self.fisher_kfac_loader = fisher_loader
        self.current_task = -1
        self.num_samples_expectation = num_samples_expectation

        self.kfac_computer = KFACComputer(device, debug_mode, train_percent, num_samples_expectation, precision=precision)

    def to_be_fishered(self, name, module, all_param_finetuned):
        if not isinstance(module, nn.Linear) \
                and not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear) \
                and not isinstance(module, nn.MultiheadAttention):
            return False
        if f"{name}.weight" in all_param_finetuned \
                or f"{name}.bias" in all_param_finetuned:
            return True
        else:
            return False

    def to_be_fishered_layer_norm(self, name, module, all_param_finetuned):
        if not isinstance(module, nn.LayerNorm):
            return False
        if f"{name}.weight" in all_param_finetuned \
                or f"{name}.bias" in all_param_finetuned:
            return True
        else:
            return False

    def compute(self, net, head, delta_w_names, dataset, use_head=False):

        self.current_task += 1

        if self.fisher_kfac_loader is None:
            ggT, aaT, ffT, num_of_examples_ggT, num_of_examples_aaT = \
                self.kfac_computer.compute(net, head, delta_w_names, dataset, use_head)
        else:
            ggT, aaT, ffT, num_of_examples_ggT, num_of_examples_aaT = \
                self.fisher_kfac_loader.load_kfac(self.current_task)

        all_param_finetuned = list(delta_w_names)
        num_of_batches = int(self.train_percent * len(dataset.train_loader))

        orig_mode = net.visual_encoder.training
        net.visual_encoder.eval()

        assert num_of_examples_ggT == num_of_examples_aaT
        num_of_examples = num_of_examples_ggT

        set_requires_grad_to(net.visual_encoder, delta_w_names, True)

        UA, UG = {}, {}

        assert ggT.keys() == aaT.keys()

        aaT_keys = list(aaT.keys())

        for i, k in tqdm(enumerate(aaT_keys), total=len(aaT_keys), desc='SVD computation'):
            aaT_matrix, ggT_matrix = aaT[k].double(), ggT[k].double()
            UA[k] = torch.linalg.svd(aaT_matrix / num_of_examples)[0]
            UG[k] = torch.linalg.svd(ggT_matrix / num_of_examples)[0]
            if self.precision == 'fp32':
                UA[k] = UA[k].float()
                UG[k] = UG[k].float()
            elif self.precision == 'fp64':
                UA[k] = UA[k].double()
                UG[k] = UG[k].double()
            else:
                raise NotImplementedError
            del aaT[k]
            del ggT[k]

        aaT.clear()
        ggT.clear()

        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in net.visual_encoder.named_parameters() if n in delta_w_names],
            lr=0.0
        )

        forward_hooks_dict = {
            'hook_forward': hook_forward_store_inputs,
            'hook_forward_nosequence': hook_forward_store_inputs,
        }

        backward_hooks_dict = {
            'hook_backward': hook_backward_ekfac,
            'hook_backward_nosequence': hook_backward_ekfac,
        }

        for name, module in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                module.UA = UA[f"{name}.weight"]
                module.UG = UG[f"{name}.weight"]
                module.precision = self.precision
                register_hooks(name, module, forward=True, backward=True,
                               forward_hooks_dict=forward_hooks_dict,
                               backward_hooks_dict=backward_hooks_dict)

        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)

        num_of_examples = 0
        for i, data in tqdm(enumerate(get_split(dataset)),
                            total=len(get_split(dataset)),
                            desc='D computation'):

            if self.debug_mode and i > 1:
                break

            if i >= num_of_batches:
                break

            x = data[0].to(self.device)
            num_of_examples += x.shape[0]

            features = net.visual_encoder(x * fake_param)
            features = features / features.norm(dim=-1, keepdim=True)

            if use_head:
                features = head(features)

            if self.num_samples_expectation > 0:
                for s in range(self.num_samples_expectation):
                    (features * torch.randn_like(features)).sum().backward(
                        retain_graph=s < self.num_samples_expectation - 1)
            else:
                features = features.sum(0)
                for cnt_class, feat in enumerate(features):
                    fake_optim.zero_grad()
                    feat.backward(retain_graph=cnt_class < features.shape[0] - 1)

        fake_optim.zero_grad()

        D = {}

        def collect_D(name, module):
            if f"{name}.weight" in all_param_finetuned:
                D[f"{name}.weight"] = getattr(module, "grad_weight") / num_of_examples

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_D(name, module)

        # remove hooks
        for name, module in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.backward_handle.remove()
                module.grad_weight = None
                module.grad_weight_c = None
                del module.grad_weight
                del module.grad_weight_c
                del module.UA
                del module.UG
                del module.precision

        set_requires_grad_to(net.visual_encoder, delta_w_names, False)
        net.visual_encoder.train(orig_mode)

        del fake_optim

        return UA, UG, D, ffT, num_of_examples

class EKFAC_Universe_Computer(nn.Module):

    def __init__(self, device: torch.device, debug_mode,
                 universe_fisher_loader: FisherLoader,
                 task_fisher_loader: FisherLoader,
                 train_percent: float = 1.0, num_samples_expectation: int = 0,
                 precision: str = 'fp64'):

        super().__init__()

        assert 0 < train_percent <= 1.0

        self.device = device
        self.debug_mode = debug_mode
        self.train_percent = train_percent
        self.precision = precision

        self.fisher_kfac_universe_loader = universe_fisher_loader
        self.fisher_kfac_task_loader = task_fisher_loader
        self.current_task = -1
        self.num_samples_expectation = num_samples_expectation
        self.UA_universe, self.UG_universe = {}, {}
        self.kfac_computer = KFACComputer(device, debug_mode, train_percent, num_samples_expectation, precision=precision)
        

    def to_be_fishered(self, name, module, all_param_finetuned):
        if not isinstance(module, nn.Linear) \
                and not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear) \
                and not isinstance(module, nn.MultiheadAttention):
            return False
        if f"{name}.weight" in all_param_finetuned \
                or f"{name}.bias" in all_param_finetuned:
            return True
        else:
            return False

    def to_be_fishered_layer_norm(self, name, module, all_param_finetuned):
        if not isinstance(module, nn.LayerNorm):
            return False
        if f"{name}.weight" in all_param_finetuned \
                or f"{name}.bias" in all_param_finetuned:
            return True
        else:
            return False
        
    def universe_base_compute(self):
        ggT_universe, aaT_universe, ffT_universe, num_of_examples_ggT_universe, num_of_examples_aaT_universe = \
                self.fisher_kfac_universe_loader.load_kfac(0)

        assert num_of_examples_ggT_universe == num_of_examples_aaT_universe
        num_of_examples_universe = num_of_examples_ggT_universe
        

        assert ggT_universe.keys() == aaT_universe.keys()
        aaT_keys = list(aaT_universe.keys())

        for i, k in tqdm(enumerate(aaT_keys), total=len(aaT_keys), desc='SVD computation'):
            aaT_matrix_universe, ggT_matrix_universe = aaT_universe[k].double(), ggT_universe[k].double()
            self.UA_universe[k] = torch.linalg.svd(aaT_matrix_universe / num_of_examples_universe)[0]
            self.UG_universe[k] = torch.linalg.svd(ggT_matrix_universe / num_of_examples_universe)[0]
            if self.precision == 'fp32':
                self.UA_universe[k] = self.UA_universe[k].float()
                self.UG_universe[k] = self.UG_universe[k].float()
            elif self.precision == 'fp64':
                self.UA_universe[k] = self.UA_universe[k].double()
                self.UG_universe[k] = self.UG_universe[k].double()
            else:
                raise NotImplementedError
            del aaT_universe[k]
            del ggT_universe[k]

        aaT_universe.clear()
        ggT_universe.clear()
        return
    
    def compute(self, net, head, delta_w_names, dataset, use_head=False):
        if self.UA_universe == {} or self.UG_universe == {}:
            try:
                UA, UG, _, _, _ = self.fisher_kfac_universe_loader.load_ekfac(0)
                self.UA_universe = UA
                self.UG_universe = UG
            except:
                self.universe_base_compute()

        self.current_task += 1

        ggT_task, aaT_task, ffT_task, num_of_examples_ggT_task, num_of_examples_aaT_task = \
                self.fisher_kfac_task_loader.load_kfac(self.current_task)

        all_param_finetuned = list(delta_w_names)
        num_of_batches = int(self.train_percent * len(dataset.train_loader))

        orig_mode = net.visual_encoder.training
        net.visual_encoder.eval()

        set_requires_grad_to(net.visual_encoder, delta_w_names, True)


        fake_optim = torch.optim.SGD(
            params=[p for (n, p) in net.visual_encoder.named_parameters() if n in delta_w_names],
            lr=0.0
        )

        forward_hooks_dict = {
            'hook_forward': hook_forward_store_inputs,
            'hook_forward_nosequence': hook_forward_store_inputs,
        }

        backward_hooks_dict = {
            'hook_backward': hook_backward_ekfac,
            'hook_backward_nosequence': hook_backward_ekfac,
        }

        for name, module in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                module.compute_bias = True if f"{name}.bias" in all_param_finetuned else False
                module.UA = self.UA_universe[f"{name}.weight"]
                module.UG = self.UG_universe[f"{name}.weight"]
                module.precision = self.precision
                register_hooks(name, module, forward=True, backward=True,
                               forward_hooks_dict=forward_hooks_dict,
                               backward_hooks_dict=backward_hooks_dict)

        fake_param = torch.tensor([1.], requires_grad=True).to(self.device)

        num_of_examples_task = 0
        for i, data in tqdm(enumerate(get_split(dataset)),
                            total=len(get_split(dataset)),
                            desc='D computation'):

            if self.debug_mode and i > 1:
                break

            if i >= num_of_batches:
                break

            x = data[0].to(self.device)
            num_of_examples_task += x.shape[0]

            features = net.visual_encoder(x * fake_param)
            features = features / features.norm(dim=-1, keepdim=True)

            if use_head:
                features = head(features)

            if self.num_samples_expectation > 0:
                for s in range(self.num_samples_expectation):
                    (features * torch.randn_like(features)).sum().backward(
                        retain_graph=s < self.num_samples_expectation - 1)
            else:
                features = features.sum(0)
                for cnt_class, feat in enumerate(features):
                    fake_optim.zero_grad()
                    feat.backward(retain_graph=cnt_class < features.shape[0] - 1)

        fake_optim.zero_grad()

        D_task = {}

        def collect_D(name, module):
            if f"{name}.weight" in all_param_finetuned:
                D_task[f"{name}.weight"] = getattr(module, "grad_weight") / num_of_examples_task

        for (name, module) in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                collect_D(name, module)

        # remove hooks
        for name, module in net.visual_encoder.named_modules():
            if self.to_be_fishered(name, module, all_param_finetuned):
                del module.compute_bias
                module.backward_handle.remove()
                module.grad_weight = None
                module.grad_weight_c = None
                del module.grad_weight
                del module.grad_weight_c
                del module.UA
                del module.UG
                del module.precision

        set_requires_grad_to(net.visual_encoder, delta_w_names, False)
        net.visual_encoder.train(orig_mode)

        del fake_optim

        return self.UA_universe, self.UG_universe, D_task, ffT_task, num_of_examples_task

class EKFAC_Difference_Computer(EKFAC_Universe_Computer):

    def __init__(self, device: torch.device, debug_mode,
                 universe_fisher_loader: FisherLoader,
                 task_fisher_loader: FisherLoader,
                 train_percent: float = 1.0, num_samples_expectation: int = 0,
                 precision: str = 'fp64', reg_ffT: bool = True
                 ):
        
        super().__init__(device, debug_mode, universe_fisher_loader, task_fisher_loader, train_percent, num_samples_expectation, precision)
        
        self.reg_ffT = reg_ffT
        
        try:
            self.UA_universe, self.UG_universe, self.D_universe, self.ffT_universe, self.num_of_examples_universe = \
                    self.fisher_kfac_universe_loader.load_ekfac(0)
        except:
                print("Universe EKFAC not found in cache, will be computed from scratch when needed.")
                print("This will take a while...")
                print("Consider precomputing and caching the universe EKFAC to speed up training.")
                self.universe_computer = EKFAComputer(device, debug_mode,
                                                 fisher_loader=universe_fisher_loader,
                                                 train_percent=train_percent,
                                                 num_samples_expectation=num_samples_expectation,
                                                 precision=precision)




    def compute(self, net, head, delta_w_names, dataset, use_head=False, proportion_coeff=0.05,
                zero_value=0.0, diff_type='difference', norm_order=2):
        def safe_den(y: torch.Tensor, eps: float = 1e-25) -> torch.Tensor:
            eps_t = torch.as_tensor(eps, dtype=y.dtype, device=y.device)
            return torch.where(y.abs() >= eps_t, y, y + eps_t)
        self.tot_gate = 0.0
        self.compute_diff = diff_type != 'none'
        D_task = {}
        ffT_task = {}
        num_of_examples_task = 0
        if self.compute_diff:
            _, _, D_task, ffT_task, num_of_examples_task = super().compute(net, head, delta_w_names, dataset, use_head)
    
        if self.UA_universe == {} or self.UG_universe == {} or self.D_universe == {} or self.ffT_universe == {}:
            print("Computing universe EKFAC from scratch...")
            self.UA_universe, self.UG_universe, self.D_universe, self.ffT_universe, self.num_of_examples_universe = \
                self.universe_computer.compute(net, head, delta_w_names, dataset, use_head)
            del self.universe_computer


        def compute_diff_tensor(universe_tensor: torch.Tensor, task_tensor: torch.Tensor, proportion_coeff: float = 0.05) -> torch.Tensor:

            universe_tensor = universe_tensor.double()
            task_tensor = task_tensor.double()
            proportion_coeff = double(proportion_coeff)
            gate = 1.0
            if diff_type == 'difference':
                out = (universe_tensor - proportion_coeff * task_tensor) / (1 - proportion_coeff)
            elif diff_type == 'log':
                out = torch.exp((torch.log(safe_den(universe_tensor)) - proportion_coeff * torch.log(safe_den(task_tensor))) / (1 - proportion_coeff))
            elif diff_type == 'cosine':
                cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-25)
                gate = 1 - cosine_similarity(universe_tensor.reshape(-1), task_tensor.reshape(-1))
                out = universe_tensor * gate
            elif diff_type == "lp_norm":
                numerator = torch.norm(universe_tensor - task_tensor, p=norm_order)
                denominator = safe_den(torch.norm(universe_tensor, p=norm_order) + torch.norm(task_tensor, p=norm_order))
                gate = numerator / denominator
                out = gate * universe_tensor
            elif diff_type == "lp_norm_tanh":
                gate = torch.tanh(torch.norm(universe_tensor - task_tensor, p=norm_order))
                out = gate * universe_tensor
            elif diff_type == "elementwise_distance":
                numerator = torch.abs(universe_tensor - task_tensor)
                denominator = safe_den(torch.abs(universe_tensor) + torch.abs(task_tensor))
                gate = numerator / denominator
                out = gate * universe_tensor
                gate = torch.mean(gate).item()
            elif diff_type == "kl":
                kl = nn.KLDivLoss(reduction='batchmean')
                normalized_universe = universe_tensor / safe_den(torch.norm(universe_tensor, p=1))
                normalized_task = task_tensor / safe_den(torch.norm(task_tensor, p=1))
                gate = torch.tanh(kl(torch.log(safe_den(normalized_universe)), safe_den(normalized_task)))
                out = gate * universe_tensor
            elif diff_type == "battacharyya":
                normalized_universe = universe_tensor / safe_den(torch.norm(universe_tensor, p=1))
                normalized_task = task_tensor / safe_den(torch.norm(task_tensor, p=1))
                bc = torch.sum(torch.sqrt(normalized_universe * normalized_task))
                gate = 1 - bc
                out = gate * universe_tensor
            elif diff_type == "jsd":
                normalized_universe = universe_tensor / safe_den(torch.norm(universe_tensor, p=1))
                normalized_task = task_tensor / safe_den(torch.norm(task_tensor, p=1))
                m = 0.5 * (normalized_universe + normalized_task)
                kl = nn.KLDivLoss(reduction='batchmean')
                jsd = 0.5 * (kl(torch.log(safe_den(normalized_universe)), safe_den(m)) + kl(torch.log(safe_den(normalized_task)), safe_den(m)))
                gate = jsd / math.log(2)
                out = gate * universe_tensor
            elif diff_type == "histogram_intersection":
                normalized_universe = universe_tensor / safe_den(torch.norm(universe_tensor, p=1))
                normalized_task = task_tensor / safe_den(torch.norm(task_tensor, p=1))
                intersection = torch.sum(torch.min(normalized_universe, normalized_task))
                gate = 1 - intersection
                out = gate * universe_tensor
            elif diff_type == "jaccard":
                normalized_universe = universe_tensor / safe_den(torch.norm(universe_tensor, p=1))
                normalized_task = task_tensor / safe_den(torch.norm(task_tensor, p=1))
                intersection = torch.sum(torch.min(normalized_universe, normalized_task))
                union = torch.sum(torch.max(normalized_universe, normalized_task))
                jaccard = intersection / safe_den(union)
                gate = 1 - jaccard
                out = gate * universe_tensor
            elif diff_type == "none":
                out = universe_tensor
            else:
                raise ValueError(f"Diff type {diff_type} not supported.")
            
            out[out < 0.0] = zero_value
            return out, gate
            
        if not self.compute_diff:
            self.tot_gate = len(self.D_universe.keys())
            return self.D_universe, self.ffT_universe
        
        D_diff, ffT_diff = {}, {}
        for k in self.D_universe.keys():
            if k not in D_task:
                raise ValueError(f"Key {k} from universe not found in task fisher information.")
            diff_tensor, gate = compute_diff_tensor(self.D_universe[k], D_task[k], proportion_coeff)
            D_diff[k] = diff_tensor
            self.tot_gate += gate
        for k in D_task.keys():
            if k not in self.D_universe:
                raise ValueError(f"Key {k} from task not found in universe fisher information.")
        for k in self.ffT_universe.keys():
            if k not in ffT_task:
                raise ValueError(f"Key {k} from universe not found in task fisher information.")
            if self.reg_ffT:
                univ_ffT = self.ffT_universe[k].double()
                U_uni, S_uni, Vt_uni = torch.linalg.svd(univ_ffT)
                
                task_proj= U_uni.T @ ffT_task[k].double() @ Vt_uni.T
                S_uni = torch.diag_embed(S_uni)
                S_task = torch.diag_embed(torch.diagonal(task_proj))
                diff_matrix, _ = compute_diff_tensor(S_uni, S_task, proportion_coeff)
                ffT_diff[k] = (U_uni @ diff_matrix @ Vt_uni)
            else:
                ffT_diff[k] = self.ffT_universe[k]
        for k in ffT_task.keys():
            if k not in self.ffT_universe:
                raise ValueError(f"Key {k} from task not found in universe fisher information.")
            
        if self.precision == 'fp32':
            for k in D_diff.keys():
                D_diff[k] = D_diff[k].float()
            for k in ffT_diff.keys():
                ffT_diff[k] = ffT_diff[k].float()
        elif self.precision == 'fp64':
            for k in D_diff.keys():
                D_diff[k] = D_diff[k].double()
            for k in ffT_diff.keys():
                ffT_diff[k] = ffT_diff[k].double()

        return self.UA_universe, self.UG_universe, D_diff, ffT_diff, num_of_examples_task, self.tot_gate

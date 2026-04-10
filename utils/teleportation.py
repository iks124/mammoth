"""
Symmetry Teleportation for Continual Learning
==============================================
Teleports model parameters to a flatter region of the loss landscape
using the ReLU positive scaling symmetry, executed at the end of each task.

Key invariance: for layers connected by ReLU, scaling pre-activation params
by t>0 and post-activation weights by 1/t preserves the network output exactly.
By optimizing t to minimize gradient norm at the current-task loss, we find a
flatter parameterization that reduces catastrophic forgetting on past tasks.

Usage:
    Add --teleport 1 --teleport_steps 50 to any Mammoth training command.

Reference:
    Zhao et al., "Symmetry Teleportation for Accelerated Optimization", NeurIPS 2022
"""

import logging
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset


# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------

def extract_teleport_pairs_resnet(net: nn.Module) -> List[Dict]:
    """
    Extract teleportable (BN, Conv) pairs from a ResNet.

    Within each BasicBlock the path  bn1 -> ReLU -> conv2  has no branching,
    so scaling bn1.{weight,bias} by t and conv2.weight input-channels by 1/t
    preserves the block output exactly.

    The initial  net.bn1 -> ReLU -> layer1[0].conv1  is also a valid pair
    (maxpool, if present, commutes with positive scaling).
    """
    from backbone.ResNetBlock import BasicBlock

    pairs: List[Dict] = []

    # NOTE: the initial net.bn1 -> relu -> layer1[0].conv1 pair is NOT valid
    # because layer1[0]'s shortcut is identity (in_planes == planes, stride == 1),
    # so the scaled bn1 output leaks through the shortcut path uncompensated.

    # Within each BasicBlock: bn1 -> relu -> conv2
    # These ARE valid because shortcut(x) bypasses bn1 entirely.
    for name, module in net.named_modules():
        if isinstance(module, BasicBlock):
            pairs.append({
                'pre_weight_key': f'{name}.bn1.weight',
                'pre_bias_key': f'{name}.bn1.bias',
                'post_weight_key': f'{name}.conv2.weight',
                'hidden_dim': module.bn1.num_features,
                'post_is_conv': True,
                'name': f'{name}.bn1->conv2',
            })

    return pairs


def extract_teleport_pairs_mlp(net: nn.Module) -> List[Dict]:
    """
    Extract teleportable (Linear_pre, Linear_post) pairs from an MLP.

    For  Linear -> ReLU -> Linear, scaling pre.weight rows & pre.bias by t
    and post.weight columns by 1/t preserves the output.
    """
    linear_layers = [(n, m) for n, m in net.named_modules()
                     if isinstance(m, nn.Linear)]

    pairs: List[Dict] = []
    for i in range(len(linear_layers) - 1):
        pre_name, pre_mod = linear_layers[i]
        post_name, _ = linear_layers[i + 1]
        # Only valid if a ReLU sits between them (we trust the architecture)
        pairs.append({
            'pre_weight_key': f'{pre_name}.weight',
            'pre_bias_key': f'{pre_name}.bias',
            'post_weight_key': f'{post_name}.weight',
            'hidden_dim': pre_mod.out_features,
            'post_is_conv': False,
            'name': f'{pre_name}->{post_name}',
        })

    return pairs


def extract_teleport_pairs(net: nn.Module) -> List[Dict]:
    """Auto-detect backbone type and extract all valid teleportation pairs."""
    try:
        from backbone.ResNetBlock import BasicBlock
        has_basic_blocks = any(isinstance(m, BasicBlock) for m in net.modules())
    except ImportError:
        has_basic_blocks = False

    if has_basic_blocks:
        pairs = extract_teleport_pairs_resnet(net)
    else:
        pairs = extract_teleport_pairs_mlp(net)

    logging.info(f"[Teleport] Found {len(pairs)} teleportable pairs: "
                 f"{[p['name'] for p in pairs]}")
    return pairs


# ---------------------------------------------------------------------------
# Core teleportation
# ---------------------------------------------------------------------------

def _build_scaled_state(pairs, all_log_t, original_state, device):
    """Build a partial state-dict with parameters scaled by exp(log_t)."""
    modified = {}
    for pair, log_t in zip(pairs, all_log_t):
        t = torch.exp(log_t)

        modified[pair['pre_weight_key']] = \
            original_state[pair['pre_weight_key']].to(device) * t
        modified[pair['pre_bias_key']] = \
            original_state[pair['pre_bias_key']].to(device) * t

        post_w = original_state[pair['post_weight_key']].to(device)
        if pair['post_is_conv']:
            # Conv weight: [out_ch, in_ch, kH, kW]
            modified[pair['post_weight_key']] = post_w / t.view(1, -1, 1, 1)
        else:
            # Linear weight: [out_features, in_features]
            modified[pair['post_weight_key']] = post_w / t.view(1, -1)

    return modified


@torch.enable_grad()
def teleport_for_flat_minimum(net: nn.Module,
                               dataloader,
                               loss_fn,
                               n_steps: int = 50,
                               lr_t: float = 0.01,
                               reg_lambda: float = 0.1,
                               device: str = 'cuda',
                               pairs: Optional[List[Dict]] = None) -> Dict:
    """
    Teleport network parameters to a flatter minimum on the *same* loss
    level-set, using the ReLU scaling symmetry.

    Args:
        net:         backbone network (e.g. model.net in Mammoth)
        dataloader:  current-task training loader
        loss_fn:     task loss (e.g. CrossEntropyLoss)
        n_steps:     optimisation steps for the scaling factors
        lr_t:        learning rate for the Adam optimiser on log(t)
        reg_lambda:  L2 penalty on log(t) to prevent extreme scaling
        device:      torch device string
        pairs:       pre-computed pair list (auto-detected if None)

    Returns:
        dict with optimization history: grad_norm_sq, reg, total_loss, max_abs_log_t per step
    """
    history = {'grad_norm_sq': [], 'reg': [], 'total_loss': [], 'max_abs_log_t': []}

    was_training = net.training
    net.eval()                              # use fixed BN running stats

    if pairs is None:
        pairs = extract_teleport_pairs(net)
    if not pairs:
        logging.warning("[Teleport] No teleportable pairs found — skipping.")
        net.train(was_training)
        return history

    # ---------- snapshot the params we will modify ----------
    state_dict = net.state_dict()
    original_state = {}
    for pair in pairs:
        for key in (pair['pre_weight_key'], pair['pre_bias_key'],
                    pair['post_weight_key']):
            original_state[key] = state_dict[key].clone()

    # ---------- learnable scaling factors ----------
    all_log_t = [torch.zeros(p['hidden_dim'], device=device, requires_grad=True)
                 for p in pairs]
    opt_t = torch.optim.Adam(all_log_t, lr=lr_t)

    data_iter = iter(dataloader)

    for step in range(n_steps):
        # --- get a batch (loop over dataloader if exhausted) ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x = batch[0].to(device)
        y = batch[1].to(device)

        # --- build differentiable scaled params ---
        modified_state = _build_scaled_state(pairs, all_log_t,
                                             original_state, device)

        # --- forward with scaled params (output is invariant) ---
        output = torch.func.functional_call(net, modified_state, (x,))
        loss = loss_fn(output, y)

        # --- gradient of loss w.r.t. scaled params ---
        modified_params = list(modified_state.values())
        grads = torch.autograd.grad(loss, modified_params, create_graph=True)

        # --- flatness objective: minimise total gradient-norm² ---
        grad_norm_sq = sum(g.pow(2).sum() for g in grads)

        # --- regularisation: keep t close to 1 ---
        reg = sum(lt.pow(2).sum() for lt in all_log_t)

        flatness_loss = grad_norm_sq + reg_lambda * reg

        # record optimization history
        history['grad_norm_sq'].append(grad_norm_sq.item())
        history['reg'].append(reg.item())
        history['total_loss'].append(flatness_loss.item())
        history['max_abs_log_t'].append(max(lt.abs().max().item() for lt in all_log_t))

        opt_t.zero_grad()
        flatness_loss.backward()
        opt_t.step()

        if step % max(n_steps // 5, 1) == 0:
            logging.info(
                f"[Teleport] step {step:3d}/{n_steps}  "
                f"grad_norm²={grad_norm_sq.item():.4f}  "
                f"reg={reg.item():.4f}  "
                f"max|log_t|={max(lt.abs().max().item() for lt in all_log_t):.4f}"
            )

    # ---------- commit final scaling ----------
    with torch.no_grad():
        final_state = net.state_dict()
        for pair, log_t in zip(pairs, all_log_t):
            t = torch.exp(log_t)

            final_state[pair['pre_weight_key']] = \
                original_state[pair['pre_weight_key']].to(device) * t
            final_state[pair['pre_bias_key']] = \
                original_state[pair['pre_bias_key']].to(device) * t

            post_w = original_state[pair['post_weight_key']].to(device)
            if pair['post_is_conv']:
                final_state[pair['post_weight_key']] = post_w / t.view(1, -1, 1, 1)
            else:
                final_state[pair['post_weight_key']] = post_w / t.view(1, -1)

        net.load_state_dict(final_state)

    # ---------- sanity check: output invariance ----------
    net.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        x_chk = batch[0].to(device)

        out_after = net(x_chk)
        out_before = torch.func.functional_call(
            net,
            {k: v.to(device) for k, v in original_state.items()},
            (x_chk,),
        )
        diff = (out_after - out_before).abs().max().item()
        if diff > 1e-2:
            logging.warning(f"[Teleport] output changed! max diff = {diff:.6f}")
        else:
            logging.info(f"[Teleport] invariance OK  (max diff = {diff:.2e})")

    net.train(was_training)
    logging.info("[Teleport] done.")
    return history


# ---------------------------------------------------------------------------
# Teleportation memory — lightweight, method-agnostic sample store
# ---------------------------------------------------------------------------

class TeleportMemory:
    """
    Stores a small number of samples per task for use during teleportation.
    Independent of any model-specific buffer — works with any CL method.
    """

    def __init__(self, samples_per_task: int = 64):
        self.samples_per_task = samples_per_task
        self.data: List[torch.Tensor] = []   # list of (x_i,) per task
        self.targets: List[torch.Tensor] = []

    def update(self, dataloader: DataLoader) -> None:
        """Sample a fixed number of examples from the current task's loader."""
        xs, ys = [], []
        count = 0
        for batch in dataloader:
            x, y = batch[0], batch[1]
            xs.append(x)
            ys.append(y)
            count += x.size(0)
            if count >= self.samples_per_task:
                break
        xs = torch.cat(xs)[:self.samples_per_task]
        ys = torch.cat(ys)[:self.samples_per_task]
        self.data.append(xs.cpu())
        self.targets.append(ys.cpu())

    def get_dataloader(self, batch_size: int = 32) -> DataLoader:
        """Return a DataLoader over all stored samples from all tasks."""
        all_x = torch.cat(self.data)
        all_y = torch.cat(self.targets)
        ds = TensorDataset(all_x, all_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          drop_last=False)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def add_teleportation_args(parser) -> None:
    """Add teleportation-related command-line arguments."""
    group = parser.add_argument_group(
        'Teleportation arguments',
        'Arguments for symmetry teleportation (flat-minimum search at end of each task).'
    )
    group.add_argument('--teleport', type=int, default=0, choices=[0, 1],
                       help='Enable symmetry teleportation at end of each task.')
    group.add_argument('--teleport_steps', type=int, default=50,
                       help='Number of optimisation steps for the scaling factors.')
    group.add_argument('--teleport_lr', type=float, default=0.01,
                       help='Learning rate for the scaling-factor optimiser.')
    group.add_argument('--teleport_reg', type=float, default=0.1,
                       help='L2 regularisation on log(t).')
    group.add_argument('--teleport_memory_per_task', type=int, default=256,
                       help='Samples stored per task for teleportation gradient computation.')

"""
Symmetry Teleportation for Continual Learning
==============================================
Two teleportation strategies are implemented:

1. ReLU scaling symmetry (original):
   For layers connected by ReLU, scaling pre-activation params by t>0 and
   post-activation weights by 1/t preserves the network output exactly.

2. COST-style LoRA teleportation (recommended):
   Adapts Zhou et al. 2025 "Continual Optimization with Symmetry Teleportation
   for Multi-Task Learning" to the CL setting.
   - Uses LoRA (low-rank adapter) as the teleportation mechanism: θ' = θ + BA
   - Objective: L_t (loss invariance on old memory) - γ * L_g (sharpness on new task)
   - Much larger parameter space (rank × model_dim vs 512) → stronger signal
   - Merges LoRA back after teleportation → no inference overhead

Usage:
    --teleport 1 --teleport_mode lora --teleport_steps 50

References:
    Zhao et al., "Symmetry Teleportation for Accelerated Optimization", NeurIPS 2022
    Zhou et al., "Continual Optimization with Symmetry Teleportation for MTL", arXiv 2503.04046
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
                               old_dataloader,
                               new_dataloader,
                               loss_fn,
                               n_steps: int = 50,
                               lr_t: float = 0.01,
                               reg_lambda: float = 0.1,
                               device: str = 'cuda',
                               pairs: Optional[List[Dict]] = None) -> Dict:
    """
    Teleport network parameters to reduce gradient conflict between old and new
    tasks, using the ReLU scaling symmetry (output-invariant transformation).

    Objective: minimise -cosine_sim(g_old, g_new) + reg_lambda * ||log_t||²
    where g_old and g_new are the gradients of the loss (w.r.t. scaled params)
    on old-task data and new-task data respectively.  The symmetry guarantee
    ensures the network output — and therefore the loss value — is unchanged by
    the t-scaling, so no explicit loss-invariance constraint is needed.

    Args:
        net:             backbone network (e.g. model.net in Mammoth)
        old_dataloader:  loader over past-task samples (TeleportMemory tasks 0..k-1)
        new_dataloader:  loader over current-task samples (TeleportMemory task k)
        loss_fn:         task loss (e.g. CrossEntropyLoss)
        n_steps:         optimisation steps for the scaling factors
        lr_t:            learning rate for the Adam optimiser on log(t)
        reg_lambda:      L2 penalty on log(t) to prevent extreme scaling
        device:          torch device string
        pairs:           pre-computed pair list (auto-detected if None)

    Returns:
        dict with optimisation history: cos_sim, reg, total_loss, max_abs_log_t per step
    """
    history = {'cos_sim': [], 'reg': [], 'total_loss': [], 'max_abs_log_t': [],
               'mean_abs_log_t': [], 'grad_norm_log_t': []}

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

    old_iter = iter(old_dataloader)
    new_iter = iter(new_dataloader)

    for step in range(n_steps):
        # --- get batches from both old and new tasks ---
        try:
            batch_old = next(old_iter)
        except StopIteration:
            old_iter = iter(old_dataloader)
            batch_old = next(old_iter)

        try:
            batch_new = next(new_iter)
        except StopIteration:
            new_iter = iter(new_dataloader)
            batch_new = next(new_iter)

        x_old, y_old = batch_old[0].to(device), batch_old[1].to(device)
        x_new, y_new = batch_new[0].to(device), batch_new[1].to(device)

        # --- build two independent scaled states so the two forward passes
        #     have completely separate computation graphs; this avoids a
        #     diamond-shaped DAG where g_old.grad_fn and g_new.grad_fn share
        #     the same modified_params node, which can cause subtle graph
        #     freeing issues during conflict_loss.backward(). ---
        modified_state_old = _build_scaled_state(pairs, all_log_t,
                                                 original_state, device)
        modified_params_old = list(modified_state_old.values())

        modified_state_new = _build_scaled_state(pairs, all_log_t,
                                                 original_state, device)
        modified_params_new = list(modified_state_new.values())

        # --- gradients on old tasks ---
        out_old = torch.func.functional_call(net, modified_state_old, (x_old,))
        loss_old = loss_fn(out_old, y_old)
        g_old = torch.autograd.grad(loss_old, modified_params_old, create_graph=True)
        g_old_flat = torch.cat([g.reshape(-1) for g in g_old])

        # --- gradients on new task ---
        out_new = torch.func.functional_call(net, modified_state_new, (x_new,))
        loss_new = loss_fn(out_new, y_new)
        g_new = torch.autograd.grad(loss_new, modified_params_new, create_graph=True)
        g_new_flat = torch.cat([g.reshape(-1) for g in g_new])

        # --- conflict objective: maximise cosine similarity ---
        cos_sim = (g_old_flat * g_new_flat).sum() / (
            g_old_flat.norm() * g_new_flat.norm() + 1e-8)

        # --- regularisation: keep t close to 1 ---
        reg = sum(lt.pow(2).sum() for lt in all_log_t)

        conflict_loss = -cos_sim + reg_lambda * reg

        # record optimisation history (before gradient step)
        history['cos_sim'].append(cos_sim.item())
        history['reg'].append(reg.item())
        history['total_loss'].append(conflict_loss.item())
        history['max_abs_log_t'].append(max(lt.abs().max().item() for lt in all_log_t))
        history['mean_abs_log_t'].append(
            sum(lt.abs().mean().item() for lt in all_log_t) / len(all_log_t))

        opt_t.zero_grad()
        conflict_loss.backward()

        # gradient norm of log_t: key signal-strength diagnostic
        grad_norm = (sum(lt.grad.pow(2).sum().item() for lt in all_log_t)) ** 0.5
        history['grad_norm_log_t'].append(grad_norm)

        opt_t.step()

        if step % max(n_steps // 5, 1) == 0:
            logging.info(
                f"[Teleport] step {step:3d}/{n_steps}  "
                f"cos_sim={cos_sim.item():.4f}  "
                f"max|log_t|={max(lt.abs().max().item() for lt in all_log_t):.4f}  "
                f"mean|log_t|={history['mean_abs_log_t'][-1]:.4f}  "
                f"grad_norm={grad_norm:.4f}"
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
        batch = next(iter(new_dataloader))
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

    def get_old_dataloader(self, batch_size: int = 32) -> Optional[DataLoader]:
        """Return a DataLoader over all tasks except the most recent one.

        Returns None if fewer than 2 tasks have been stored (no 'old' tasks yet).
        """
        if len(self.data) < 2:
            return None
        all_x = torch.cat(self.data[:-1])
        all_y = torch.cat(self.targets[:-1])
        ds = TensorDataset(all_x, all_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          drop_last=False)

    def get_new_dataloader(self, batch_size: int = 32) -> DataLoader:
        """Return a DataLoader over the most recently stored task's samples."""
        ds = TensorDataset(self.data[-1], self.targets[-1])
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          drop_last=False)


# ---------------------------------------------------------------------------
# COST-style LoRA Teleportation for CL
# ---------------------------------------------------------------------------

def _get_lora_params(net: nn.Module, rank: int, device: str,
                     target_types=(nn.Linear, nn.Conv2d)):
    """
    Create LoRA A, B matrices for all Linear and Conv2d layers in the network.

    Returns a list of dicts with keys:
        weight_key, B, A, type, shape (for conv)
    """
    lora_list = []
    state = net.state_dict()

    for name, module in net.named_modules():
        if not isinstance(module, target_types):
            continue
        weight_key = f'{name}.weight'
        if weight_key not in state:
            continue

        w = state[weight_key]
        if isinstance(module, nn.Linear):
            out_f, in_f = w.shape
            B = torch.zeros(out_f, rank, device=device, requires_grad=True)
            A = torch.randn(rank, in_f, device=device) * 0.01
            A = A.detach().requires_grad_(True)
            lora_list.append({'weight_key': weight_key, 'B': B, 'A': A,
                               'type': 'linear'})
        elif isinstance(module, nn.Conv2d):
            out_ch, in_ch, kH, kW = w.shape
            B = torch.zeros(out_ch, rank, device=device, requires_grad=True)
            A = torch.randn(rank, in_ch * kH * kW, device=device) * 0.01
            A = A.detach().requires_grad_(True)
            lora_list.append({'weight_key': weight_key, 'B': B, 'A': A,
                               'type': 'conv',
                               'shape': (out_ch, in_ch, kH, kW)})

    return lora_list


def _build_lora_state(lora_list, original_state, device):
    """Build partial state-dict with LoRA perturbation applied."""
    modified = {}
    for lp in lora_list:
        delta = lp['B'] @ lp['A']  # (out_f, in_f) or (out_ch, in_ch*kH*kW)
        if lp['type'] == 'conv':
            delta = delta.reshape(*lp['shape'])
        modified[lp['weight_key']] = original_state[lp['weight_key']].to(device) + delta
    return modified


@torch.enable_grad()
def teleport_lora_for_cl(net: nn.Module,
                          old_dataloader,
                          new_dataloader,
                          loss_fn,
                          n_steps: int = 50,
                          lr_lora: float = 1e-3,
                          gamma: float = 1.0,
                          lora_rank: int = 4,
                          sharpness_radius: float = 0.05,
                          n_sharpness: int = 5,
                          device: str = 'cuda') -> dict:
    """
    COST-style LoRA teleportation adapted for Continual Learning.

    Objective (per step):
        L_lora = L_t  -  gamma * L_g

        L_t  = mean_{i in old_tasks} |L_i(θ+ΔΘ) - L_i(θ)|
               (loss invariance: teleportation must not hurt old tasks)

        L_g  = max_{j=1..n_sharpness} L_new(θ+ΔΘ + ε_j), ε_j ~ Sphere(radius)
               (sharpness proxy: find a sharper, more convergent point for new task)

    After optimisation, LoRA weights BA are merged into the network.
    The network output on old tasks is approximately preserved (up to L_t ≈ 0).

    Args:
        net:               backbone network
        old_dataloader:    loader over past-task memory (tasks 0..t-1)
        new_dataloader:    loader over current-task samples (task t)
        loss_fn:           task loss (e.g. CrossEntropyLoss)
        n_steps:           LoRA optimisation steps
        lr_lora:           learning rate for Adam on LoRA params
        gamma:             weight for sharpness objective vs loss invariance
        lora_rank:         rank r for LoRA adapters
        sharpness_radius:  radius δ of perturbation sphere for sharpness
        n_sharpness:       number of random perturbations to sample
        device:            torch device string

    Returns:
        dict with per-step history: l_t, l_g, total_loss
    """
    history = {'l_t': [], 'l_g': [], 'total_loss': [], 'delta_norm': []}

    was_training = net.training
    net.eval()

    # Snapshot original weights
    original_state = {k: v.clone() for k, v in net.state_dict().items()}

    # Compute frozen reference losses on old tasks (L*_i in COST paper)
    with torch.no_grad():
        old_ref_losses = []
        for batch in old_dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = net(x)
            old_ref_losses.append(loss_fn(out, y).item())
        l_old_ref = sum(old_ref_losses) / len(old_ref_losses)

    # Build LoRA parameters
    lora_list = _get_lora_params(net, rank=lora_rank, device=device)
    if not lora_list:
        logging.warning("[Teleport-LoRA] No LoRA-able layers found — skipping.")
        net.train(was_training)
        return history

    all_lora = [lp['B'] for lp in lora_list] + [lp['A'] for lp in lora_list]
    opt_lora = torch.optim.Adam(all_lora, lr=lr_lora)

    n_lora_params = sum(p.numel() for p in all_lora)
    logging.info(f"[Teleport-LoRA] {len(lora_list)} layers, "
                 f"rank={lora_rank}, total LoRA params={n_lora_params}")

    old_iter = iter(old_dataloader)
    new_iter = iter(new_dataloader)

    for step in range(n_steps):
        # Refresh iterators
        try:
            batch_old = next(old_iter)
        except StopIteration:
            old_iter = iter(old_dataloader)
            batch_old = next(old_iter)
        try:
            batch_new = next(new_iter)
        except StopIteration:
            new_iter = iter(new_dataloader)
            batch_new = next(new_iter)

        x_old, y_old = batch_old[0].to(device), batch_old[1].to(device)
        x_new, y_new = batch_new[0].to(device), batch_new[1].to(device)

        # Build teleported state
        lora_state = _build_lora_state(lora_list, original_state, device)

        # --- L_t: Loss invariance on old tasks ---
        out_old = torch.func.functional_call(net, lora_state, (x_old,))
        l_old_new = loss_fn(out_old, y_old)
        l_t = torch.abs(l_old_new - l_old_ref)

        # --- L_g: Sharpness proxy on new task ---
        # Sample n_sharpness random perturbations from sphere of radius δ
        # Add to ALL LoRA-modified weights (reuse lora_state base)
        sharpness_vals = []
        for _ in range(n_sharpness):
            perturbed_state = {}
            for key, w in lora_state.items():
                eps = torch.randn_like(w)
                eps = eps * (sharpness_radius / (eps.norm() + 1e-8))
                perturbed_state[key] = w + eps
            out_new_p = torch.func.functional_call(net, perturbed_state, (x_new,))
            sharpness_vals.append(loss_fn(out_new_p, y_new))

        l_g = torch.stack(sharpness_vals).max()

        total_loss = l_t - gamma * l_g

        history['l_t'].append(l_t.item())
        history['l_g'].append(l_g.item())
        history['total_loss'].append(total_loss.item())

        # delta norm (how far LoRA moves the weights)
        with torch.no_grad():
            dn = sum((lp['B'] @ lp['A']).norm().item() for lp in lora_list)
        history['delta_norm'].append(dn)

        opt_lora.zero_grad()
        total_loss.backward()
        opt_lora.step()

        if step % max(n_steps // 5, 1) == 0:
            logging.info(
                f"[Teleport-LoRA] step {step:3d}/{n_steps}  "
                f"L_t={l_t.item():.4f}  L_g={l_g.item():.4f}  "
                f"delta_norm={dn:.4f}"
            )

    # --- Merge LoRA into network weights ---
    with torch.no_grad():
        final_state = net.state_dict()
        for lp in lora_list:
            delta = lp['B'] @ lp['A']
            if lp['type'] == 'conv':
                delta = delta.reshape(*lp['shape'])
            final_state[lp['weight_key']] = (
                original_state[lp['weight_key']].to(device) + delta.detach()
            )
        net.load_state_dict(final_state)

    # --- Sanity check: loss on old tasks should be approximately preserved ---
    net.eval()
    with torch.no_grad():
        post_losses = []
        for batch in old_dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            post_losses.append(loss_fn(net(x), y).item())
        l_old_post = sum(post_losses) / len(post_losses)
        logging.info(
            f"[Teleport-LoRA] done. Old-task loss: "
            f"{l_old_ref:.4f} → {l_old_post:.4f} "
            f"(delta={l_old_post - l_old_ref:+.4f})"
        )

    net.train(was_training)
    return history


def apply_htr(optimizer, delta_theta: torch.Tensor, grad_pre: torch.Tensor):
    """
    Historical Trajectory Reuse (HTR) from COST paper.

    Modulates the Adam optimizer's momentum state after teleportation based
    on the alignment between the teleportation direction and the pre-teleport
    gradient. This prevents stale momentum from misleading post-teleport training.

    σ = cos_sim(Δθ, g')
    v_t ← σ β₁ v_{t-1} + (1 - σ β₁) g_t   (first moment)
    s_t ← σ β₂ s_{t-1} + (1 - σ β₂) g_t²   (second moment)

    Args:
        optimizer:    the model's Adam-style optimizer (modifies in-place)
        delta_theta:  flattened teleportation displacement (θ' - θ)
        grad_pre:     flattened gradient at pre-teleportation point
    """
    sigma = F.cosine_similarity(
        delta_theta.unsqueeze(0), grad_pre.unsqueeze(0)
    ).clamp(-1, 1).item()

    for group in optimizer.param_groups:
        beta1 = group.get('betas', (0.9, 0.999))[0]
        beta2 = group.get('betas', (0.9, 0.999))[1]
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].mul_(sigma * beta1)
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].mul_(sigma * beta2)

    logging.info(f"[Teleport-HTR] σ={sigma:.3f} — momentum modulated.")


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
                       help='L2 regularisation on log(t) (used in scaling mode only).')
    group.add_argument('--teleport_memory_per_task', type=int, default=256,
                       help='Samples stored per task for teleportation gradient computation.')
    group.add_argument('--teleport_mode', type=str, default='scaling',
                       choices=['scaling', 'lora'],
                       help='Teleportation mode: "scaling" (ReLU symmetry) or "lora" (COST-style).')
    group.add_argument('--teleport_lora_rank', type=int, default=4,
                       help='LoRA rank for COST-style teleportation.')
    group.add_argument('--teleport_gamma', type=float, default=1.0,
                       help='Weight γ for sharpness objective in LoRA mode.')
    group.add_argument('--teleport_sharpness_radius', type=float, default=0.05,
                       help='Perturbation radius δ for sharpness estimation in LoRA mode.')
    group.add_argument('--teleport_htr', type=int, default=0, choices=[0, 1],
                       help='Enable Historical Trajectory Reuse (HTR) after LoRA teleportation.')

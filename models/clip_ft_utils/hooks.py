import torch
from torch.nn import functional as F


def hook_forward_batch_nosequence(module, inputs, _):
    def concatone(x):
        return torch.cat((x, torch.ones_like(torch.sum(x, -1, keepdim=True))), -1)

    with torch.no_grad():
        if module.precision == "fp32":
            x = inputs[0].detach().float()
        elif module.precision == "fp64":
            x = inputs[0].detach().double()
        else:
            raise NotImplementedError

        assert len(x.shape) == 2

        if hasattr(module, "bias") and module.compute_bias:
            x = concatone(x)

        x = x.T @ x
        if hasattr(module, "gram_input"):
            module.gram_input += x
        else:
            setattr(module, "gram_input", x)


def hook_forward_batch(module, inputs, _):
    def concatone(x):
        return torch.cat((x, torch.ones_like(torch.sum(x, -1, keepdim=True))), -1)

    with torch.no_grad():
        if module.precision == "fp32":
            x = inputs[0].detach().float()
        elif module.precision == "fp64":
            x = inputs[0].detach().double()
        else:
            raise NotImplementedError

        assert len(x.shape) == 3

        if hasattr(module, "bias") and module.compute_bias:
            x = concatone(x)

        if "attn" in module.name:
            _, r, c = x.shape
        else:
            r, _, c = x.shape

        x = x.reshape(-1, c)
        x = x.T @ ((1.0 / r) * x)
        if hasattr(module, "gram_input"):
            module.gram_input += x
        else:
            setattr(module, "gram_input", x)


def hook_forward_store_inputs(module, inputs, _):
    setattr(module, "inputs", inputs[0])


@torch.no_grad()
def hook_backward_nosequence(module, _, grad_output):
    if module.precision == "fp32":
        grad_out = grad_output[0].float()
    elif module.precision == "fp64":
        grad_out = grad_output[0].double()
    else:
        raise NotImplementedError

    if len(grad_out.shape) == 2:
        grad_weight = grad_out.T @ grad_out
    else:
        raise NotImplementedError

    if not hasattr(module, "gram_grad"):
        module.gram_grad = grad_weight.clone()
        module.gram_grad_c = torch.zeros_like(grad_weight)
    else:
        y = grad_weight - module.gram_grad_c
        t = module.gram_grad + y
        module.gram_grad_c = (t - module.gram_grad) - y
        module.gram_grad = t


@torch.no_grad()
def hook_backward(module, _, grad_output):
    if module.precision == "fp32":
        grad_out = grad_output[0].float()
    elif module.precision == "fp64":
        grad_out = grad_output[0].double()
    else:
        raise NotImplementedError

    if len(grad_out.shape) > 2:
        if "attn.proj" in module.name or "attn.qkv" in module.name:
            _, r, c = grad_out.shape
        else:
            r, _, c = grad_out.shape
        grad_out = grad_out.reshape(-1, c)
        grad_weight = grad_out.T @ ((1 / r) * grad_out)
    else:
        raise NotImplementedError

    if not hasattr(module, "gram_grad"):
        module.gram_grad = grad_weight.clone()
        module.gram_grad_c = torch.zeros_like(grad_weight)
    else:
        y = grad_weight - module.gram_grad_c
        t = module.gram_grad + y
        module.gram_grad_c = (t - module.gram_grad) - y
        module.gram_grad = t


@torch.no_grad()
def hook_backward_cls_token(module, _, grad_output):
    if module.precision == "fp32":
        grad_out = grad_output[0].float()
    elif module.precision == "fp64":
        grad_out = grad_output[0].double()
    else:
        raise NotImplementedError

    gram_grad = grad_out[:, 0].T @ grad_out[:, 0]

    if not hasattr(module, "gram_grad"):
        module.gram_grad = gram_grad.clone()
        module.gram_grad_c = torch.zeros_like(gram_grad)
    else:
        y = gram_grad - module.gram_grad_c
        t = module.gram_grad + y
        module.gram_grad_c = (t - module.gram_grad) - y
        module.gram_grad = t


@torch.no_grad()
def hook_backward_layer_norm(module, _, grad_output):
    if module.precision == "fp32":
        grad_out = grad_output[0].float()
        inputs = module.inputs.float()
        normalized = F.layer_norm(inputs, module.normalized_shape).float()
    elif module.precision == "fp64":
        grad_out = grad_output[0].double()
        inputs = module.inputs.double()
        normalized = F.layer_norm(inputs, module.normalized_shape).double()
    else:
        raise NotImplementedError

    grad_weight = grad_out * normalized

    if len(grad_out.shape) > 2:
        if "ln_pre" in module.name:
            grad_weight = grad_weight.sum(1)
        else:
            grad_weight = grad_weight.sum(0)

    batch_gram_w = grad_weight.T @ grad_weight

    if not hasattr(module, "gram_grad_weight"):
        module.gram_grad_weight = torch.zeros_like(batch_gram_w)
        module.gram_grad_weight_c = torch.zeros_like(batch_gram_w)

    y_w = batch_gram_w - module.gram_grad_weight_c
    t_w = module.gram_grad_weight + y_w
    module.gram_grad_weight_c = (t_w - module.gram_grad_weight) - y_w
    module.gram_grad_weight = t_w

    if hasattr(module, "bias") and module.compute_bias:
        grad_bias = grad_out
        if len(grad_out.shape) > 2:
            if "ln_pre" in module.name:
                grad_bias = grad_bias.sum(1)
            else:
                grad_bias = grad_bias.sum(0)

        batch_gram_b = grad_bias.T @ grad_bias

        if not hasattr(module, "gram_grad_bias"):
            module.gram_grad_bias = torch.zeros_like(batch_gram_b)
            module.gram_grad_bias_c = torch.zeros_like(batch_gram_b)

        y_b = batch_gram_b - module.gram_grad_bias_c
        t_b = module.gram_grad_bias + y_b
        module.gram_grad_bias_c = (t_b - module.gram_grad_bias) - y_b
        module.gram_grad_bias = t_b

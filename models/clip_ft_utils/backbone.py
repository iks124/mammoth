from copy import deepcopy
import types

import torch
import torch.nn as nn
from backbone import MammothBackbone

try:
    import clip
    from clip.model import CLIP
except ImportError as exc:
    raise ImportError(
        "Please install CLIP: pip install git+https://github.com/openai/CLIP.git"
    ) from exc

from models.clip_ft_utils.templates import get_templates


def create_clip(name_clip_backbone, device) -> CLIP:
    clip_model, _ = clip.load(name_clip_backbone, device=torch.device("cpu"), jit=False)
    surgery(clip_model)
    return clip_model.to(device)


@torch.no_grad()
def surgery(clip_model: CLIP):
    num_blocks = len(clip_model.visual.transformer.resblocks)
    embed_dim = clip_model.visual.class_embedding.shape[0]
    for block_id in range(num_blocks):
        old_ma = clip_model.visual.transformer.resblocks[block_id].attn
        old_ma_sd = old_ma.state_dict()
        new_ma = MultiheadAttention(embed_dim, old_ma.num_heads, True).to("cpu")
        new_ma.qkv.weight.zero_()
        new_ma.qkv.weight.add_(old_ma_sd["in_proj_weight"])
        new_ma.qkv.bias.zero_()
        new_ma.qkv.bias.add_(old_ma_sd["in_proj_bias"])
        new_ma.proj.weight.zero_()
        new_ma.proj.weight.add_(old_ma_sd["out_proj.weight"])
        new_ma.proj.bias.zero_()
        new_ma.proj.bias.add_(old_ma_sd["out_proj.bias"])
        del clip_model.visual.transformer.resblocks[block_id].attn
        clip_model.visual.transformer.resblocks[block_id].attn = new_ma
    replace_visual_outproj(clip_model)


class ClsEmbedder(nn.Module):
    def __init__(self, class_embedding: torch.Tensor):
        super().__init__()
        self.register_parameter(
            "class_embedding", nn.Parameter(class_embedding.clone())
        )

    def forward(self, x: torch.Tensor):
        return torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )


def get_custom_forward(old_forward):
    def custom_visual_forward(ext, x: torch.Tensor):
        x = ext.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = ext.cls_token_layer(x)
        x = x + ext.positional_embedding.to(x.dtype)
        x = ext.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = ext.transformer(x)
        x = x.permute(1, 0, 2)
        x = ext.ln_post(x[:, 0, :])
        x = ext.lin_proj(x)
        return x

    return custom_visual_forward


def replace_visual_outproj(clip_model):
    visual_proj: torch.Tensor = clip_model.visual.proj.clone()
    clip_model.visual.lin_proj = nn.Linear(
        visual_proj.shape[1], visual_proj.shape[0], bias=False
    )
    clip_model.visual.lin_proj.weight = nn.Parameter(visual_proj.T)
    clip_model.visual.lin_proj.requires_grad_(clip_model.visual.proj.requires_grad)
    clip_model.visual.proj = None
    del clip_model.visual.proj
    old_forward = clip_model.visual.forward.__func__
    cls_token_layer = ClsEmbedder(
        clip_model.visual.class_embedding.clone()
    ).requires_grad_(True)
    clip_model.visual.register_module("cls_token_layer", cls_token_layer)
    clip_model.visual.class_embedding = None
    del clip_model.visual.class_embedding
    clip_model.visual.forward = types.MethodType(
        get_custom_forward(old_forward), clip_model.visual
    )


class MultiheadAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        n, b, c = query.shape
        query = query.transpose(0, 1)
        qkv = self.qkv(query)
        qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 0)
        return x, attn


def build_classification_head(clip_model, dataset, offset, eval=False, all_heads=False):
    template = get_templates(dataset.NAME)
    classnames = dataset.get_class_names()
    device = clip_model.text_projection.device
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [
                t(classname) if callable(t) else t.format(classname) for t in template
            ]
            text_tokens = clip.tokenize(texts).to(device)
            embeddings = clip_model.encode_text(text_tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= 100.0
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    if all_heads:
        classification_head = ClassificationHead(
            normalize=True, weights=zeroshot_weights
        )
    else:
        if eval:
            classification_head = ClassificationHead(
                normalize=True, weights=zeroshot_weights[:][: offset[1]]
            )
        else:
            classification_head = ClassificationHead(
                normalize=True, weights=zeroshot_weights[:][offset[0] : offset[1]]
            )
    classification_head.requires_grad_(False)
    return classification_head


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = nn.Parameter(biases.clone())
        else:
            self.bias = nn.Parameter(
                torch.zeros_like(self.bias, device=self.weight.device)
            )

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)


class Backbone(MammothBackbone):
    @torch.no_grad()
    def __init__(self, clip_model, dataset, args) -> None:
        super().__init__()
        self.dataset = dataset
        self.dtype = torch.float32
        self.args = args
        self.visual_encoder = deepcopy(clip_model.to(dtype=torch.float32).visual)
        self.copy_visual_encoder(clip_model)
        self.classes = self.dataset.get_class_names()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )
        self.task_id = 0

    def copy_visual_encoder(self, clip_model):
        self.visual_encoder.load_state_dict(clip_model.visual.state_dict())

    def forward(self, x):
        image_features = self.visual_encoder(x.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

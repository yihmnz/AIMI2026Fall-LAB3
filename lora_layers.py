"""LoRA on weights: works with fused QKV and MultiheadAttention functional paths."""
import math
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint


class LowRankWeight(nn.Module):
    def __init__(self, weight, rank, alpha, qv=False):
        super().__init__()
        rows, cols = weight.shape
        if rank < 1 or rank > min(rows, cols):
            raise ValueError('Invalid LoRA rank')
        self.scale = alpha / rank
        self.qv = qv
        if qv and rows != 3 * cols:
            raise ValueError('Expected packed [Q;K;V] weight')
        self.A = nn.Parameter(weight.new_empty((2 if qv else 1, rank, cols)))
        self.B = nn.Parameter(weight.new_zeros((2 if qv else 1, cols if qv else rows, rank)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, weight):
        delta = self.B @ self.A
        if self.qv:
            delta = torch.cat((delta[0], torch.zeros_like(delta[0]), delta[1]), dim=0)
        else:
            delta = delta[0]
        return weight + delta * self.scale


def inject(model, branch, rank=8, alpha=16, last_blocks=4):
    """Freeze base; adapt Q and V only. Return exact registered parameter targets."""
    model.requires_grad_(False)
    candidates = []
    if branch == 'raddino':
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(('.attention.attention.query', '.attention.attention.value',
                                                               '.attention.q_proj', '.attention.v_proj')):
                candidates.append((name, module, 'weight', False))
        # Two projections in each HF DINOv2 block.
        keep = 2 * last_blocks if last_blocks else len(candidates)
    elif branch == 'chexfound':
        candidates = [(n, m, 'weight', True) for n, m in model.named_modules()
                      if isinstance(m, nn.Linear) and n.endswith('.attn.qkv')]
        keep = last_blocks or len(candidates)
    elif branch == 'chexzero':
        candidates = [(n, m, 'in_proj_weight', True) for n, m in model.named_modules()
                      if n.startswith('visual.') and isinstance(m, nn.MultiheadAttention)]
        keep = last_blocks or len(candidates)
    else:
        raise ValueError(branch)
    if not candidates or (last_blocks and keep > len(candidates)):
        raise RuntimeError(f'Unexpected {branch} attention API; found {len(candidates)} targets')
    targets = []
    for name, module, attr, qv in candidates[-keep:]:
        parametrize.register_parametrization(module, attr, LowRankWeight(getattr(module, attr), rank, alpha, qv))
        targets.append(name + '.' + attr)
    return targets


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_adapter(model, state):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(expected):
        raise ValueError('Adapter parameter names do not match this model/target configuration')
    with torch.no_grad():
        for name, p in expected.items():
            if state[name].shape != p.shape or not torch.isfinite(state[name]).all():
                raise ValueError(f'Invalid adapter tensor: {name}')
            p.copy_(state[name])


def use_sdpa(model):
    """Replace CheXFound's materialized NxN attention with native PyTorch SDPA."""
    count = 0
    for module in model.modules():
        if not (hasattr(module, 'qkv') and hasattr(module, 'num_heads') and hasattr(module, 'proj_drop')):
            continue
        original = module.forward
        def make_forward(fallback):
            def forward(self, x, attn_bias=None, return_attention=False):
                if return_attention:
                    return fallback(x, return_attention=True)
                if attn_bias is not None or not torch.is_tensor(x):
                    raise ValueError('LoRA runner supports dense image batches, no nested attention bias')
                b, n, c = x.shape
                q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
                z = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
                return self.proj_drop(self.proj(z.transpose(1, 2).reshape(b, n, c)))
            return forward
        module.forward = types.MethodType(make_forward(original), module)
        count += 1
    if not count:
        raise RuntimeError('No CheXFound attention modules found')
    return count


def checkpoint_blocks(model, branch):
    if branch == 'raddino':
        blocks = model.encoder.layer
    elif branch == 'chexfound':
        if model.chunked_blocks:
            raise ValueError('Build CheXFound with block_chunks=0')
        blocks = model.blocks
    else:
        blocks = model.visual.transformer.resblocks
    for block in blocks:
        original = block.forward
        def make_forward(fn):
            def forward(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(fn, *args, use_reentrant=False, **kwargs)
                return fn(*args, **kwargs)
            return forward
        block.forward = make_forward(original)

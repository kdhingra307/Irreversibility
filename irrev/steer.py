"""Activation steering: add a direction to the residual stream during generation.

Directions are found from contrast pairs at one layer, then added at EVERY position
during generation -- the standard recipe (Turner et al.; Arditi et al.). Coefficients
are expressed as multiples of the typical activation norm at that layer, so a
coefficient of 1.0 means "a perturbation the size of the activations themselves"
rather than an uninterpretable raw magnitude.

Steering runs in BOTH directions on purpose. One-directional steering cannot
distinguish a specific feature from generic caution: if adding a vector raises
confirmation-seeking, so might any perturbation. The stronger test is that the
positive coefficient raises confirmation-seeking on irreversible actions AND the
negative coefficient suppresses it, including on actions that genuinely warrant it.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def diff_of_means(A, y, layer):
    """Arditi-style direction: mean(class 1) - mean(class 0) at one layer."""
    X = A[:, layer, :]
    return unit(X[np.asarray(y) == 1].mean(0) - X[np.asarray(y) == 0].mean(0))


def layer_norm_scale(A, layer):
    """Typical activation norm at a layer, used to make coefficients comparable."""
    return float(np.linalg.norm(A[:, layer, :], axis=-1).mean())


def matched_random(direction, seed=0):
    """Random orientation, same norm. The 'any perturbation does this' control."""
    rng = np.random.default_rng(seed)
    r = rng.normal(size=len(direction)).astype(np.float32)
    return unit(r)


def _blocks(model):
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate transformer blocks")


@contextlib.contextmanager
def steering(model, direction, coeff, layers, scale=1.0):
    """Add coeff*scale*direction to the residual stream at activation index `layers`.

    INDEXING -- verified empirically, not assumed. `acts.cache` reads
    `output_hidden_states`, a tuple of length n_layers+1 where

        hidden_states[0] = embedding output
        hidden_states[L] = output of block L-1 = INPUT to block L

    The first implementation used a forward hook on `blocks[L]`, which was wrong
    twice over: it targeted the output of block L (activation index L+1), and a
    forward hook's replacement value turns out not to be what gets recorded at that
    index at all -- measured, the change first appears at L+2.

    So this uses a forward PRE-hook on `blocks[L]`, which rewrites the tensor
    entering block L. That tensor is exactly hidden_states[L], so a direction read
    at activation index L is injected at activation index L. `layers` is therefore
    in the same index space as `acts.cache` and `probes.*` throughout.

    Valid indices are 0..n_layers-1. Index n_layers is the post-final-norm output,
    after every block, so there is nothing left to condition on it.

    coeff=0 installs no hooks at all, so the zero condition is bit-identical to an
    unsteered run rather than merely numerically close.
    """
    if coeff == 0:
        yield
        return
    blocks = _blocks(model)
    bad = [L for L in layers if L < 0 or L >= len(blocks)]
    if bad:
        raise ValueError(
            f"activation indices {bad} out of range: expected 0..{len(blocks) - 1} "
            f"(index {len(blocks)} is the post-final-norm output and has no block "
            f"downstream of it)")
    dev = next(model.parameters()).device
    vec = torch.tensor(np.asarray(direction, dtype=np.float32), device=dev) * (coeff * scale)
    handles = []

    def pre_hook(_module, args, kwargs):
        if args:
            h = args[0]
            return (h + vec.to(h.dtype),) + args[1:], kwargs
        if "hidden_states" in kwargs:
            h = kwargs["hidden_states"]
            kwargs = {**kwargs, "hidden_states": h + vec.to(h.dtype)}
        return args, kwargs

    try:
        for act_idx in layers:
            handles.append(
                blocks[act_idx].register_forward_pre_hook(pre_hook, with_kwargs=True))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate_steered(tok, model, prompts, direction, coeff, layers, scale,
                     max_new_tokens=128, batch=4):
    outs = []
    with steering(model, direction, coeff, layers, scale):
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(next(model.parameters()).device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for seq, inp in zip(gen, enc["input_ids"]):
                outs.append(tok.decode(seq[len(inp):], skip_special_tokens=True))
    return outs

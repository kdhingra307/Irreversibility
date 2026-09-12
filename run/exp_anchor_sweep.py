"""Probe-position sweep: capture at chosen anchors, not just the final token.

Loads data/jed_anchor_dataset.jsonl and rebuilds nothing. Anchor indices were computed
and stored at build time, so nothing here re-derives them.

WHY HOOKS INSTEAD OF output_hidden_states
Every earlier run used `output_hidden_states=True`, which materialises the hidden state
of EVERY token at all 37 layers and then keeps one. At 2100 tokens that is ~400 MB per
sequence, and over this session it drove the machine into swap: 11 of 12 GB swap in use,
the model's weights paged out, and throughput collapsing from 2.3 to 9.2 s/prompt.
Forward hooks that slice only the anchor positions cut the retained tensor by ~1000x.

THE INDEXING TRAP, AND HOW IT IS CHECKED
`output_hidden_states` returns 37 entries for a 36-layer model:
    (embeddings, out_L0, ..., out_L34, norm(out_L35))
so the LAST entry is post-final-norm and the raw out_L35 never appears. A hook on
`layers[i]` yields out_Li, which equals hidden_states[i+1] for i <= 34 but NOT for i=35.
Getting this wrong is exactly the off-by-one that invalidated Experiment 3 earlier, so
`validate()` compares hook capture against output_hidden_states on real rows and refuses
to run if any layer disagrees beyond float tolerance.

Batch size is 1 throughout: anchors are ABSOLUTE token indices and left padding would
shift every one of them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from irrev.acts import load
from irrev.probes import cv_auc, make_probe

ANCHORS = ["A1", "A2"]


def capture(model, ids, positions):
    """Hidden state at each position, every layer. Returns (n_layers+1, n_pos, hidden).

    Layout matches output_hidden_states: row 0 is the embedding output, row i+1 is the
    output of block i, and the final row has the model's final norm applied.
    """
    blocks = model.model.layers
    out = {}
    handles = []

    def mk(i):
        def hook(_m, _a, o):
            h = o[0] if isinstance(o, tuple) else o
            out[i] = h[0, positions, :].detach().float().cpu()
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(
        lambda _m, _a, o: out.__setitem__(-1, o[0, positions, :].detach().float().cpu())))
    for i, b in enumerate(blocks):
        handles.append(b.register_forward_hook(mk(i)))
    handles.append(model.model.norm.register_forward_hook(
        lambda _m, _a, o: out.__setitem__("norm", o[0, positions, :].detach().float().cpu())))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    n = len(blocks)
    rows = [out[-1]] + [out[i] for i in range(n - 1)] + [out["norm"]]
    return torch.stack(rows).numpy()


def validate(model, tok, prompts, positions_list, atol=2e-2):
    """Hook capture must equal output_hidden_states, layer by layer."""
    for p, pos in zip(prompts, positions_list):
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            next(model.parameters()).device)
        got = capture(model, ids, pos)
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        want = np.stack([h[0, pos, :].float().cpu().numpy() for h in hs])
        bad = [i for i in range(want.shape[0])
               if not np.allclose(got[i], want[i], atol=atol)]
        if bad:
            d = [float(np.abs(got[i] - want[i]).max()) for i in bad]
            raise AssertionError(f"hook capture disagrees at layers {bad}, max abs {d}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/jed_anchor_dataset.jsonl")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--tag", default="anchor")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    LOG = open(f"data/{a.tag}_log.txt", "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    S = [json.loads(l) for l in open(a.dataset)]
    if a.limit:
        S = S[:a.limit]
    D = pd.DataFrame(S)
    say(f"{len(D)} rows | {D.cue.value_counts().to_dict()} | "
        f"tokens {D.n_tokens.min()}-{D.n_tokens.max()}")

    tok, model = load(a.model)
    NL, HID = model.config.num_hidden_layers, model.config.hidden_size
    DEV = next(model.parameters()).device

    say("\nvalidating hook capture against output_hidden_states on 2 rows...")
    validate(model, tok, [S[0]["prompt"], S[-1]["prompt"]],
             [[S[0]["anchor_A1"], S[0]["anchor_A2"]],
              [S[-1]["anchor_A1"], S[-1]["anchor_A2"]]])
    say("  hook capture matches at every layer")

    CACHE = f"data/{a.tag}_acts.npy"
    if os.path.exists(CACHE) and np.load(CACHE, mmap_mode="r").shape[0] == len(D):
        A = np.load(CACHE); say(f"loaded cached activations {A.shape}")
    else:
        A = np.zeros((len(D), NL + 1, len(ANCHORS), HID), dtype=np.float32)
        t0 = time.time()
        for i, r in enumerate(S):
            ids = tok(r["prompt"], return_tensors="pt",
                      add_special_tokens=False)["input_ids"].to(DEV)
            A[i] = capture(model, ids, [r["anchor_A1"], r["anchor_A2"]])
            if i % 60 == 0:
                say(f"  captured {i}/{len(D)} ({time.time()-t0:.0f}s)")
        np.save(CACHE, A); say(f"captured {A.shape} in {time.time()-t0:.0f}s")

    y, g = D.y.values, D.doc.values
    R = {}
    say("\n" + "=" * 84)
    say("WITHIN CUE CONDITION, per anchor (grouped CV by document)")
    say("=" * 84)
    say(f"  {'condition':28} {'anchor':>7} {'n':>5} {'L1':>6} {'min':>6} {'peak':>6} {'@L':>4}")
    for cue in ("structural", "declarative"):
        idx = np.where((D.cue == cue).values)[0]
        for ai, an in enumerate(ANCHORS):
            c = np.array([cv_auc(A[idx][:, l, ai, :], y[idx], g[idx], n_splits=8)
                          for l in range(NL + 1)])
            pk = int(np.nanargmax(c))
            R[f"{cue}|{an}"] = {"n": len(idx), "layer1": float(c[1]), "peak": float(c[pk]),
                                "min": float(np.nanmin(c)), "peak_layer": pk,
                                "curve": c.tolist()}
            say(f"  {cue:28} {an:>7} {len(idx):5} {c[1]:6.3f} {np.nanmin(c):6.3f} "
                f"{c[pk]:6.3f} {pk:4}")

    say("\n" + "=" * 84)
    say("TRANSFER per anchor -- layer chosen by CV on TRAIN, then scored once")
    say("=" * 84)
    say(f"  {'direction':30} {'anchor':>7} {'layer':>6} {'AUC':>7} {'(argmax-on-test)':>18}")
    for tr_cue in ("declarative", "structural"):
        te_cue = "structural" if tr_cue == "declarative" else "declarative"
        tr = np.where((D.cue == tr_cue).values)[0]
        te = np.where((D.cue == te_cue).values)[0]
        for ai, an in enumerate(ANCHORS):
            trcv = np.array([cv_auc(A[tr][:, l, ai, :], y[tr], g[tr], n_splits=8)
                             for l in range(NL + 1)])
            L = int(np.nanargmax(trcv))
            curve = np.array([roc_auc_score(y[te], make_probe()
                              .fit(A[tr, l, ai, :], y[tr])
                              .predict_proba(A[te, l, ai, :])[:, 1])
                              for l in range(NL + 1)])
            k = f"{tr_cue}->{te_cue}|{an}"
            R[k] = {"layer_from_train": L, "auc": float(curve[L]),
                    "auc_argmax_on_test": float(np.nanmax(curve)),
                    "argmax_layer": int(np.nanargmax(curve)),
                    "min": float(np.nanmin(curve)), "curve": curve.tolist()}
            say(f"  {tr_cue+' -> '+te_cue:30} {an:>7} {L:6} {curve[L]:7.3f} "
                f"{np.nanmax(curve):18.3f}")

    D.drop(columns=["prompt", "listing", "listing_cue", "toolbox"], errors="ignore").to_csv(
        f"data/{a.tag}_rows.csv", index=False)
    json.dump(R, open(f"data/{a.tag}.json", "w"), indent=1, default=float)
    say(f"\nwrote data/{a.tag}.json, data/{a.tag}_rows.csv")
    say("DONE")
    LOG.close()


if __name__ == "__main__":
    main()

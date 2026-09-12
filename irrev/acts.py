"""Residual-stream activation caching at the decision point.

"The last token before the tool call" is ambiguous, and the two readings answer
different questions, so both are built:

  ANTICIPATORY -- the context ends at the assistant generation prompt. The model
    has the tools, the request and the prior observations, but has not named an
    action. Probing here asks whether the model represents the irreversibility of
    the action it is *about to* take. This is the deployment-relevant point and
    the one steering must act on; it is also PING's final-token readout.

  EVALUATIVE -- the proposed call is appended as the opening of the assistant
    turn. Probing here asks whether the model represents *this named action* as
    irreversible. Cleaner signal, weaker claim.

A probe that works only in the evaluative frame is reading the tool name it was
just handed, not anticipating a consequence.
"""
from __future__ import annotations

import json

import numpy as np
import torch

MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",              # 36 layers, 2560 hidden -- primary
    "qwen3-8b": "Qwen/Qwen3-8B",              # 36 layers, 4096 hidden -- scale rung
    "qwen2.5-7b": "qwen/qwen2.5-7b-instruct",  # 28 layers, 3584 hidden -- earlier runs
}
MODEL_ID = MODELS["qwen3-4b"]

# Absolute layer indices do NOT transfer between these: layer 18 is 64% depth on
# Qwen2.5-7B and 50% on Qwen3. Always report relative depth and re-find the peak layer.


def resolve(model_id: str) -> str:
    """Accept either a short key from MODELS or a full HF id."""
    return MODELS.get(model_id.lower(), model_id)


def load(model_id: str = MODEL_ID, device: str | None = None, dtype=torch.float16):
    """Load a model for probing.

    `device=None` picks mps > cuda > cpu. dtype defaults to float16 because MPS
    handles bfloat16 poorly, even though these checkpoints are bf16-native.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = resolve(model_id)
    if device is None:
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # keeps the last token at index -1
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device).eval()
    return tok, model


def _template(tok, msgs, tools, thinking: bool | None):
    """apply_chat_template, passing enable_thinking only to models that accept it.

    Qwen3 changes the generation prompt by mode:
        thinking=True   ...<|im_start|>assistant\\n            -> next token opens <think>
        thinking=False  ...assistant\\n<think>\\n\\n</think>\\n\\n  -> next token is the action
    Only `thinking=False` is structurally comparable to a non-reasoning model, because
    only there is the probed token immediately followed by the action.
    """
    kw = dict(tools=tools, tokenize=False, add_generation_prompt=True)
    if thinking is not None:
        try:
            return tok.apply_chat_template(msgs, enable_thinking=thinking, **kw)
        except TypeError:
            pass          # template predates the flag (e.g. Qwen2.5)
    return tok.apply_chat_template(msgs, **kw)


def render(tok, stim, frame: str = "anticipatory", thinking: bool | None = None,
           continuation: str | None = None) -> str:
    """Build the prompt.

    frame="anticipatory"  ends at the assistant turn; probe the final token.
    frame="evaluative"    appends a tool call; probe its last token.

    `continuation` overrides the appended call in the evaluative frame. This is what
    T3 (paired probing) uses: the context is held byte-identical and the appended
    action becomes the manipulation, so a probe fit on it cannot be reading context.
    It is also how the on-policy variant injects the model's OWN emitted call rather
    than a hand-written one.
    """
    sys_txt = stim.system
    if stim.prior:
        sys_txt += "\n\nRecent tool observations:\n" + stim.prior
    msgs = [{"role": "system", "content": sys_txt},
            {"role": "user", "content": stim.user}]
    text = _template(tok, msgs, stim.tools, thinking)
    if frame == "evaluative":
        if continuation is None:
            continuation = '{"name": "%s", "arguments": %s}' % (
                stim.call_name, json.dumps(stim.call_args, sort_keys=True))
        text += "<tool_call>\n" + continuation
    return text


@torch.no_grad()
def cache(tok, model, stims, frame="anticipatory", batch_size=4, device=None,
          verbose=True, thinking: bool | None = None, continuations=None):
    """Return (acts, meta) with acts of shape [n_stim, n_layer+1, hidden].

    Reads the LAST token of each prompt. Note the indexing convention that
    `steer.steering` must invert: hidden_states[0] is the embedding output and
    hidden_states[L] is the output of block L-1.

    `continuations` is an optional per-stimulus list used by T3 paired probing.
    """
    if device is None:
        device = next(model.parameters()).device
    conts = continuations if continuations is not None else [None] * len(stims)
    texts = [render(tok, s, frame, thinking=thinking, continuation=c)
             for s, c in zip(stims, conts)]
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    out = np.zeros((len(stims), n_layers + 1, hidden), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for li, h in enumerate(hs):                      # h: [b, t, hidden]
            out[i:i + len(chunk), li, :] = h[:, -1, :].float().cpu().numpy()
        if verbose and (i // batch_size) % 8 == 0:
            print(f"  cached {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
    meta = {"frame": frame, "n_layers": n_layers, "hidden": hidden, "thinking": thinking,
            "token_lengths": [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]}
    return out, meta

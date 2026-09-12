"""Stimulus set v3 -- decoys token-matched EXACTLY, verb templates diversified.

v2 held the tool COUNT constant, which cut length leakage from AUC 1.000 to 0.516.
Two residual problems remained:

  * matching was approximate (mean |diff| 2.2 tokens over the tool schema), so the
    two prompts of a pair still differed slightly in length;
  * all 16 decoys were named `summarize_*` with a `*_prefix` parameter, while every
    recovery tool used an undo verb with an `_id` parameter. That is a naming
    template, and a probe could read it instead of reading recoverability.

v3 fixes both. Each decoy's tool JSON is tuned (see scratchpad/tune2.py) to match
its recovery tool's token count exactly, so within a reversibility pair the two
prompts are identical in length. Decoy verbs are spread over compare / count /
export / chart / profile / tally / benchmark rather than all being `summarize`.

The decoys live in data/decoys_tuned.json so the tuning is auditable and the
token match can be re-verified rather than trusted.
"""
from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path

from .stimuli import LEXICAL_IRREV, LEXICAL_REV, Stim
from .stimuli2 import D2

_DECOYS_PATH = Path(__file__).resolve().parent.parent / "data" / "decoys_tuned.json"


def load_decoys(path=None) -> dict:
    return json.loads(Path(path or _DECOYS_PATH).read_text())


def build3(decoys: dict | None = None) -> list[Stim]:
    """128 contexts; within each pair the two prompts match in token length exactly."""
    dec = decoys or load_decoys()
    out: list[Stim] = []
    for d in D2:
        for harm, irrev, stk in product((0, 1), (0, 1), (0, 1)):
            fourth = dec[d.key] if irrev else d.recovery
            out.append(Stim(
                uid=f"{d.key}|h{harm}|i{irrev}|s{stk}",
                domain=d.key,
                pair_id=f"{d.key}|h{harm}|s{stk}",
                harm=harm, irreversible=irrev, stakes=stk,
                system=d.system,
                user=(d.req_harmful if harm else d.req_benign),
                prior=f"{d.prior}\n{d.stakes['high' if stk else 'low']}",
                tools=[d.target] + d.support + [fourth],
                call_name=d.target["function"]["name"],
                call_args=(d.call_harmful if harm else d.call_benign),
            ))
    return out


def build3_lexical(decoys: dict | None = None) -> list[Stim]:
    """Cross-decode arm: the decoy sits in BOTH arms, so no recovery tool is ever
    present and no naming template distinguishes the arms. Only a sentence in the
    target tool's description moves. A probe that transfers here is not reading
    tool vocabulary."""
    dec = decoys or load_decoys()
    out: list[Stim] = []
    for d in D2:
        for harm, irrev, stk in product((0, 1), (0, 1), (0, 1)):
            target = copy.deepcopy(d.target)
            target["function"]["description"] += (LEXICAL_IRREV if irrev else LEXICAL_REV)
            out.append(Stim(
                uid=f"LEX|{d.key}|h{harm}|i{irrev}|s{stk}",
                domain=d.key, pair_id=f"LEX|{d.key}|h{harm}|s{stk}",
                harm=harm, irreversible=irrev, stakes=stk,
                system=d.system,
                user=(d.req_harmful if harm else d.req_benign),
                prior=f"{d.prior}\n{d.stakes['high' if stk else 'low']}",
                tools=[target] + d.support + [dec[d.key]],
                call_name=d.target["function"]["name"],
                call_args=(d.call_harmful if harm else d.call_benign),
            ))
    return out


def check_pair_lengths(stims, tok, frame="anticipatory") -> dict:
    """Per-pair prompt-length equality -- the invariant v3 exists to guarantee.

    Distribution overlap is a weak check; two arms can overlap in aggregate while
    every individual pair still differs. This asserts the difference pair by pair.
    """
    from collections import defaultdict

    from .acts import render
    n = lambda s: len(tok(render(tok, s, frame), add_special_tokens=False)["input_ids"])
    by_pair = defaultdict(dict)
    for s in stims:
        by_pair[s.pair_id][s.irreversible] = n(s)
    diffs = {p: v[1] - v[0] for p, v in by_pair.items() if 0 in v and 1 in v}
    return {
        "n_pairs": len(diffs),
        "n_exact": sum(1 for d in diffs.values() if d == 0),
        "max_abs_diff": max((abs(d) for d in diffs.values()), default=0),
        "offenders": {p: d for p, d in diffs.items() if d != 0},
    }

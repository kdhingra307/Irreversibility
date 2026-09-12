"""Programmatic invariant checks on the stimulus set. Gates, not inspections."""
from __future__ import annotations

import json
from collections import defaultdict

from .stimuli import LEXICAL_BLOCKLIST


def _text(s):
    """Everything the model actually sees, as one string."""
    return "\n".join([s.system, s.user, s.prior, json.dumps(s.tools), s.call_str()]).lower()


def check_blocklist(stims):
    """No stimulus may lexically declare (ir)reversibility."""
    bad = []
    for s in stims:
        t = _text(s)
        for w in LEXICAL_BLOCKLIST:
            if w in t:
                bad.append((s.uid, w))
    return bad


def check_pair_invariants(stims):
    """Within a reversibility pair: the proposed call and the request must be
    byte-identical, and the tool inventory must differ by exactly the one
    recovery tool."""
    by_pair = defaultdict(list)
    for s in stims:
        by_pair[s.pair_id].append(s)
    viol = []
    for pid, members in by_pair.items():
        if len(members) != 2:
            viol.append((pid, f"expected 2 members, got {len(members)}")); continue
        a, b = sorted(members, key=lambda s: s.irreversible)   # a=reversible
        if a.call_str() != b.call_str():
            viol.append((pid, "proposed call differs across reversibility axis"))
        if (a.system, a.user, a.prior) != (b.system, b.user, b.prior):
            viol.append((pid, "prose differs across reversibility axis"))
        na = {t["function"]["name"] for t in a.tools}
        nb = {t["function"]["name"] for t in b.tools}
        if not nb < na:
            viol.append((pid, "irreversible arm's inventory is not a strict subset"))
        elif len(na - nb) != 1:
            viol.append((pid, f"inventory differs by {len(na-nb)} tools, expected 1"))
    return viol


def check_harm_invariants(stims):
    """Across the harm axis, holding domain/irrev/stakes fixed, the tool
    inventory must be identical -- harm must not leak through the tool surface."""
    by = defaultdict(list)
    for s in stims:
        by[(s.domain, s.irreversible, s.stakes)].append(s)
    viol = []
    for k, members in by.items():
        if len(members) != 2:
            viol.append((k, f"expected 2 harm variants, got {len(members)}")); continue
        a, b = sorted(members, key=lambda s: s.harm)
        if json.dumps(a.tools, sort_keys=True) != json.dumps(b.tools, sort_keys=True):
            viol.append((k, "tool inventory differs across harm axis"))
        if a.system != b.system:
            viol.append((k, "system prompt differs across harm axis"))
    return viol


def check_stakes_invariants(stims):
    """Across the stakes axis, only the prior observation may move."""
    by = defaultdict(list)
    for s in stims:
        by[(s.domain, s.harm, s.irreversible)].append(s)
    viol = []
    for k, members in by.items():
        a, b = sorted(members, key=lambda s: s.stakes)
        if (a.system, a.user, a.call_str()) != (b.system, b.user, b.call_str()):
            viol.append((k, "something other than the prior moved across stakes axis"))
        if json.dumps(a.tools, sort_keys=True) != json.dumps(b.tools, sort_keys=True):
            viol.append((k, "tool inventory differs across stakes axis"))
    return viol


def run_all(stims):
    return {
        "lexical_blocklist": check_blocklist(stims),
        "reversibility_pairs": check_pair_invariants(stims),
        "harm_axis": check_harm_invariants(stims),
        "stakes_axis": check_stakes_invariants(stims),
    }


def check_pair_invariants_v2(stims):
    """v2: both arms carry the same tool COUNT, so the inventories must differ by
    exactly one tool swapped in each direction (symmetric difference of 2)."""
    from collections import defaultdict
    by_pair = defaultdict(list)
    for s in stims:
        by_pair[s.pair_id].append(s)
    viol = []
    for pid, members in by_pair.items():
        if len(members) != 2:
            viol.append((pid, f"expected 2 members, got {len(members)}")); continue
        a, b = sorted(members, key=lambda s: s.irreversible)
        if a.call_str() != b.call_str():
            viol.append((pid, "proposed call differs across reversibility axis"))
        if (a.system, a.user, a.prior) != (b.system, b.user, b.prior):
            viol.append((pid, "prose differs across reversibility axis"))
        na = {t["function"]["name"] for t in a.tools}
        nb = {t["function"]["name"] for t in b.tools}
        if len(na) != len(nb):
            viol.append((pid, f"tool COUNT differs ({len(na)} vs {len(nb)}) - length confound"))
        elif len(na ^ nb) != 2:
            viol.append((pid, f"inventories differ by {len(na ^ nb)} names, expected 2"))
    return viol


def run_all_v2(stims):
    return {
        "lexical_blocklist": check_blocklist(stims),
        "reversibility_pairs": check_pair_invariants_v2(stims),
        "harm_axis": check_harm_invariants(stims),
        "stakes_axis": check_stakes_invariants(stims),
    }


def check_pair_invariants_v4(stims):
    """v4: reversibility lives in the ACTION. Across the reversibility axis the tool
    NAME set must be identical (no inventory manipulation at all), the proposed call
    and all prose must be identical, and the ONLY textual difference permitted is the
    target tool's own description."""
    import json as _json
    from collections import defaultdict
    by_pair = defaultdict(list)
    for s in stims:
        by_pair[s.pair_id].append(s)
    viol = []
    for pid, members in by_pair.items():
        if len(members) != 2:
            viol.append((pid, f"expected 2 members, got {len(members)}")); continue
        a, b = sorted(members, key=lambda s: s.irreversible)   # a = reversible
        if {t["function"]["name"] for t in a.tools} != {t["function"]["name"] for t in b.tools}:
            viol.append((pid, "tool NAME sets differ - affordance confound is back"))
        if a.call_str() != b.call_str():
            viol.append((pid, "proposed call differs"))
        if (a.system, a.user, a.prior) != (b.system, b.user, b.prior):
            viol.append((pid, "prose differs"))
        # every non-target tool must be byte-identical
        for x, y in zip(a.tools[1:], b.tools[1:]):
            if _json.dumps(x, sort_keys=True) != _json.dumps(y, sort_keys=True):
                viol.append((pid, f"support tool {x['function']['name']} differs"))
        ta, tb = a.tools[0]["function"], b.tools[0]["function"]
        if ta["name"] != tb["name"]:
            viol.append((pid, "target tool name differs"))
        if _json.dumps(ta["parameters"], sort_keys=True) != _json.dumps(tb["parameters"], sort_keys=True):
            viol.append((pid, "target parameter schema differs"))
        if ta["description"] == tb["description"]:
            viol.append((pid, "target description identical - no manipulation applied"))
    return viol


def check_no_recovery_tool(stims):
    """Neither arm may expose a recovery tool: agent affordance is held constant
    (absent) so that intrinsic reversibility is the only thing moving."""
    import re
    PAT = (r"(restore|undo|revert|rollback|recover|reinstate|unarchive|reopen|"
           r"reactivate|undelete|unlock|unblock|unmute|unassign|unsubscribe|"
           r"unflag|regrant|recall|rescind|unpublish|reverse|cancel)")
    bad = []
    for s in stims:
        # skip tools[0]: that is the target action itself. `cancel_event` is the
        # action under consideration, not a tool that could undo it.
        for t in s.tools[1:]:
            if re.search(PAT, t["function"]["name"]):
                bad.append((s.uid, t["function"]["name"]))
    return bad


def run_all_v4(stims):
    return {
        "lexical_blocklist": check_blocklist(stims),
        "reversibility_pairs": check_pair_invariants_v4(stims),
        "harm_axis": check_harm_invariants(stims),
        "stakes_axis": check_stakes_invariants(stims),
        "no_recovery_tool_in_scope": check_no_recovery_tool(stims),
    }

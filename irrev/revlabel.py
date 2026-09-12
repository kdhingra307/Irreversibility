"""Derive reversibility labels for AgentAbstain commit tools from source structure.

AgentAbstain labels tools lookup / verify / commit -- i.e. *whether* state changes,
never whether the change can be undone. This module recovers the missing axis by
reading what each commit tool actually does to state, and whether the environment
exposes a semantic inverse. Output follows Magentic-UI's three-level ActionGuard
taxonomy: never / maybe / always irreversible.

The labels are evidence-bearing, not oracles: every tool carries the cues that
produced its label so a human can audit and override.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- what the body does to state -------------------------------------------------
DESTRUCTIVE = [
    (r"\bdel\s+self\.state", "del-on-state"),
    (r"\.pop\(", "pop"),
    (r"\.remove\(", "remove"),
    (r"\.clear\(\)", "clear"),
    (r"=\s*\[[^\]]*for\s+\w+\s+in\s+self\.state", "filter-reassign"),
]
ADDITIVE = [(r"\.append\(", "append"), (r"\.extend\(", "extend"),
            (r"\.insert\(", "insert")]
FLAGGING = [(r'\.status\s*=\s*["\']', "status-flag"),
            (r'\[["\']status["\']\]\s*=', "status-key"),
            (r"\.(is_|has_)\w+\s*=\s*(True|False)", "bool-flag"),
            (r"\.deleted\s*=\s*True", "tombstone")]

# Emission verbs: the effect leaves the sandbox, so no state edit can retract it.
EMISSION = r"(send|submit|post|publish|dispatch|transfer|pay|wire|notify|email|sms|" \
           r"broadcast|escalate|file_|report_|order|book|purchase|charge|refund|" \
           r"deploy|execute_trade|place_|issue_|dial|call_)"
# Verb pairs that make a tool undoable *if the partner exists in the same env*.
INVERSE_PAIRS = [
    ("create", "delete"), ("create", "remove"), ("add", "remove"), ("add", "delete"),
    ("set", "unset"), ("set", "clear"), ("enable", "disable"), ("lock", "unlock"),
    ("assign", "unassign"), ("grant", "revoke"), ("open", "close"),
    ("subscribe", "unsubscribe"), ("archive", "restore"), ("freeze", "unfreeze"),
    ("activate", "deactivate"), ("schedule", "cancel"), ("book", "cancel"),
    ("apply", "revert"), ("start", "stop"), ("attach", "detach"),
    ("approve", "reject"), ("check_in", "check_out"), ("mute", "unmute"),
    ("suspend", "reinstate"), ("block", "unblock"), ("flag", "unflag"),
]
RESTORE_HINT = r"(restore|undo|revert|rollback|recover|reinstate|unarchive|reopen|" \
               r"reactivate|un(delete|do|lock|block|mute|assign|subscribe))"


@dataclass
class RevLabel:
    qualname: str
    kind: str
    level: str                       # never | maybe | always (irreversible)
    effect: str                      # additive | flagging | destructive | emission | unknown
    inverse_tool: str | None = None
    cues: list[str] = field(default_factory=list)
    confidence: str = "low"


def _hits(patterns, text):
    return [tag for pat, tag in patterns if re.search(pat, text)]


def _find_inverse(name: str, env_tools: set[str]) -> str | None:
    """Look for a same-environment tool that semantically undoes `name`."""
    for a, b in INVERSE_PAIRS:
        for fwd, inv in ((a, b), (b, a)):
            if name.startswith(fwd + "_") or f"_{fwd}_" in name or name.endswith("_" + fwd):
                cand = name.replace(fwd, inv, 1)
                if cand != name and cand in env_tools:
                    return cand
    # generic restore-ish tool naming the same object
    obj = re.sub(r"^(create|add|set|delete|remove|update|send|post|submit|cancel)_", "", name)
    for t in env_tools:
        if t != name and re.search(RESTORE_HINT, t) and obj and obj in t:
            return t
    return None


def label_tool(tool, kind: str, env_tools: set[str]) -> RevLabel:
    body = tool.body_src
    name = tool.name
    cues: list[str] = []

    if kind in ("lookup", "verify"):
        return RevLabel(tool.qualname, kind, "never", "read-only",
                        cues=["kind=" + kind], confidence="high")

    d, a, f = _hits(DESTRUCTIVE, body), _hits(ADDITIVE, body), _hits(FLAGGING, body)
    emits = bool(re.search(r"^" + EMISSION, name)) or bool(re.search(r"\b" + EMISSION, name))
    inverse = _find_inverse(name, env_tools)

    if emits:
        effect, level, conf = "emission", "always", "high"
        cues.append("emission-verb:" + name.split("_")[0])
    elif d:
        effect, level, conf = "destructive", "always", "high"
        cues += ["destructive:" + x for x in d]
    elif f and not a:
        effect = "flagging"
        cues += ["flag:" + x for x in f]
        level, conf = ("maybe", "medium") if inverse is None else ("never", "medium")
    elif a:
        effect = "additive"
        cues += ["additive:" + x for x in a]
        level, conf = ("never", "medium") if inverse else ("maybe", "medium")
    else:
        effect, level, conf = "unknown", "maybe", "low"
        cues.append("no-structural-cue")

    if inverse:
        cues.append("inverse:" + inverse)
        # an explicit inverse downgrades everything except emissions, which
        # cannot be recalled once they leave the sandbox
        if effect != "emission":
            level = "never" if effect in ("additive", "flagging") else "maybe"
            conf = "medium"

    return RevLabel(tool.qualname, kind, level, effect, inverse, cues, conf)


def label_all(tools, kinds: dict[str, str]) -> list[RevLabel]:
    by_env: dict[str, set[str]] = {}
    for t in tools:
        by_env.setdefault(t.env, set()).add(t.name)
    return [label_tool(t, kinds.get(t.qualname, "unknown"), by_env[t.env]) for t in tools]

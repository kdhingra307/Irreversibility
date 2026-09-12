"""Harvest the AgentAbstain tool surface by AST parsing, without running MCP servers.

The 42 environments each define their tools as FastMCP-decorated inner functions
inside `BaseEnvironment._register_tools`. We recover name, signature, docstring
and body source for every one of them.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Tool:
    env: str
    name: str
    qualname: str          # "<env>.<name>", matching the DAG node `tool` field
    doc: str
    params: list[dict] = field(default_factory=list)
    returns: str = ""
    body_src: str = ""
    lineno: int = 0

    @property
    def n_required(self) -> int:
        return sum(1 for p in self.params if not p["has_default"])


def _ann(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _is_mcp_tool(fn: ast.FunctionDef) -> bool:
    """True for functions carrying an @self.mcp.tool(...) decorator."""
    for d in fn.decorator_list:
        call = d.func if isinstance(d, ast.Call) else d
        if isinstance(call, ast.Attribute) and call.attr == "tool":
            owner = call.value
            if isinstance(owner, ast.Attribute) and owner.attr == "mcp":
                return True
    return False


def parse_env(env_dir: Path) -> list[Tool]:
    src_path = env_dir / "environment.py"
    if not src_path.exists():
        return []
    src = src_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src)
    lines = src.splitlines()
    tools: list[Tool] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not _is_mcp_tool(node):
            continue
        a = node.args
        defaults = list(a.defaults)
        pos = list(a.posonlyargs) + list(a.args)
        # align defaults to the tail of positional args
        pad = [None] * (len(pos) - len(defaults))
        paired = list(zip(pos, pad + defaults))
        params = []
        for arg, dflt in paired:
            if arg.arg in ("self", "cls"):
                continue
            params.append({
                "name": arg.arg,
                "annotation": _ann(arg.annotation),
                "has_default": dflt is not None,
                "default": _ann(dflt) if dflt is not None else None,
            })
        for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
            params.append({
                "name": arg.arg,
                "annotation": _ann(arg.annotation),
                "has_default": dflt is not None,
                "default": _ann(dflt) if dflt is not None else None,
            })
        body_src = "\n".join(lines[node.lineno - 1: node.end_lineno])
        tools.append(Tool(
            env=env_dir.name,
            name=node.name,
            qualname=f"{env_dir.name}.{node.name}",
            doc=(ast.get_docstring(node) or "").strip(),
            params=params,
            returns=_ann(node.returns),
            body_src=body_src,
            lineno=node.lineno,
        ))
    return tools


def harvest(data_dir: str | Path) -> list[Tool]:
    root = Path(data_dir) / "environments"
    out: list[Tool] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("__"):
            out.extend(parse_env(d))
    return out


def dag_kinds(tasks_jsonl: str | Path) -> dict[str, str]:
    """Map tool qualname -> kind (lookup/verify/commit) from the task DAGs.

    These are the benchmark's own labels; only `commit` tools enter the
    critical-action set for operational tasks.
    """
    kinds: dict[str, set] = {}
    for line in open(tasks_jsonl):
        row = json.loads(line)
        dag = row.get("execution_dag") or {}
        for n in dag.get("nodes") or []:
            t, k = n.get("tool"), n.get("kind")
            if t and k:
                kinds.setdefault(t, set()).add(k)
    # collapse; flag any tool the DAGs label inconsistently
    return {t: (ks.pop() if len(ks) == 1 else "AMBIG:" + "|".join(sorted(ks)))
            for t, ks in kinds.items()}


def to_records(tools: list[Tool]) -> list[dict]:
    recs = []
    for t in tools:
        d = asdict(t)
        d["n_params"] = len(t.params)
        d["n_required"] = t.n_required
        recs.append(d)
    return recs


def env_class_maps(env_dir: Path) -> dict:
    """Pull the class-level `tool_kinds` / `mutation_tools` declarations.

    `tool_kinds` is the benchmark's authoritative partition over ALL tools in the
    environment (lookup / verify / commit); the task DAGs only exercise a subset,
    so this is the label source to prefer.
    """
    src_path = env_dir / "environment.py"
    if not src_path.exists():
        return {}
    tree = ast.parse(src_path.read_text(encoding="utf-8", errors="ignore"))
    out = {"tool_kinds": {}, "mutation_tools": [], "mutation_id_fields": []}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for stmt in cls.body:
            targets = []
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                targets = [stmt.target.id]
                val = stmt.value
            elif isinstance(stmt, ast.Assign):
                targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                val = stmt.value
            else:
                continue
            for name in targets:
                if name not in out or val is None:
                    continue
                try:
                    lit = ast.literal_eval(val)
                except Exception:
                    continue
                if name == "tool_kinds" and isinstance(lit, dict):
                    out["tool_kinds"].update(lit)
                elif name in ("mutation_tools", "mutation_id_fields"):
                    out[name] = sorted(lit)
    return out


def kind_table(data_dir: str | Path) -> dict[str, str]:
    """qualname -> kind, from every environment's own `tool_kinds` declaration."""
    root = Path(data_dir) / "environments"
    table = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("__"):
            for tname, kind in env_class_maps(d).get("tool_kinds", {}).items():
                table[f"{d.name}.{tname}"] = kind
    return table

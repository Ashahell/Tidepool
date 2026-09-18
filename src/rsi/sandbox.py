# src/rsi/sandbox.py
from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor

ALLOWED_IMPORTS = {"math", "json", "random", "itertools", "collections"}
ALLOWED_BUILTINS = {n: b for n, b in __builtins__.items()} if isinstance(__builtins__, dict) else {}
SAFE_BUILTINS = {k: ALLOWED_BUILTINS[k] for k in
    ["abs", "min", "max", "range", "len", "int", "float", "str",
     "bool", "list", "dict", "set", "tuple", "sorted", "enumerate",
     "zip", "round", "sum"] if k in ALLOWED_BUILTINS}
BANNED_ATTRS = {"__dict__", "__class__", "__subclasses__", "globals",
                "locals", "eval", "exec", "open", "compile", "__import__"}

class PolicyRejected(Exception):
    pass

def check_policy_source(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PolicyRejected(f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise PolicyRejected(f"banned import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise PolicyRejected(f"banned import: {node.module}")
        elif isinstance(node, ast.While):
            raise PolicyRejected("while loops are banned; use bounded for")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                raise PolicyRejected(f"banned attribute: {node.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pass
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in funcs:
        for child in ast.walk(fn):
            if child is not fn and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                raise PolicyRejected("nested functions are banned")

def load_policy(code: str, base_cls):
    check_policy_source(code)
    # __build_class__ is interpreter machinery required to execute any
    # `class` statement; not callable-useful to candidates under AST rules.
    _ns_builtins = dict(SAFE_BUILTINS)
    _ns_builtins.setdefault("__build_class__", __build_class__)  # noqa: F821
    ns = {"__builtins__": _ns_builtins, "__name__": "<candidate>",
            base_cls.__name__: base_cls}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)
    except Exception as e:
        raise PolicyRejected(f"exec failed: {e}")
    cand = ns.get("CandidatePolicy")
    if not (isinstance(cand, type) and issubclass(cand, base_cls)):
        raise PolicyRejected("module must define CandidatePolicy(base)")
    try:
        return cand()
    except Exception as e:
        raise PolicyRejected(f"instantiation failed: {e}")

def run_with_timeout(fn, timeout: float):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        return fut.result(timeout=timeout)

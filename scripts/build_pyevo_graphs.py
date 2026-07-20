"""
build_pyevo_graphs.py — Heterogeneous API Evolution Graph Builder for PyEvo

Constructs a heterogeneous graph G = (V_api ∪ V_code, E) for each sample,
covering both the API evolution history (Evolution-Aware View) and the
surrounding code context (Code-Context View). A 3-hop BFS prunes nodes
irrelevant to the target API call site.

Usage: python scripts/build_pyevo_graphs.py --output_dir ./data/pyevo_graphs
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dsets import PyEvoDataset
from util.globals import DATA_DIR


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def _clean_sig(sig: str) -> str:
    """Strip ellipsis placeholders from API signatures."""
    if not sig:
        return sig
    sig = re.sub(r',\s*\.\.\.\s*\)', ')', sig)   # func(a, b, ...) -> func(a, b)
    sig = re.sub(r'\(\s*\.\.\.\s*,\s*', '(', sig)  # func(..., b) -> func(b)
    sig = sig.replace('(...)', '()')               # func(...) -> func()
    return sig.strip()


# ---------------------------------------------------------------------------
# Signature parsing
# ---------------------------------------------------------------------------

def parse_python_signature(sig: str) -> Dict[str, Any]:
    """
    Parse a Python function signature (with type annotations).

    Returns dict with keys: name, async_, params, return_type, is_method.
    Example input:  def foo(x: int, y: str = "hello") -> bool
    Example output: {
        'name': 'foo',
        'async_': False,
        'params': [{'name': 'x', 'type': 'int', 'default': None}, ...],
        'return_type': 'bool',
        'is_method': False,
    }
    """
    result: Dict[str, Any] = {
        'name': '',
        'async_': False,
        'params': [],
        'return_type': None,
        'is_method': False,
    }
    if not sig or not sig.strip():
        return result

    sig = sig.strip()
    if sig.startswith('async '):
        result['async_'] = True
        sig = sig[6:].strip()

    name_match = re.search(r'\bdef\s+(\w+)\s*\(', sig)
    if name_match:
        result['name'] = name_match.group(1)

    # Return type: -> ... (before trailing colon if present)
    ret_match = re.search(r'\)\s*->\s*(.+?)(?:\s*:\s*$|\s*$)', sig)
    if ret_match:
        result['return_type'] = ret_match.group(1).strip().rstrip(':').strip()

    # Params section: text between the first '(' and last ')'
    open_idx = sig.find('(')
    close_idx = sig.rfind(')')
    if open_idx != -1 and close_idx > open_idx:
        params_str = sig[open_idx + 1: close_idx].strip()
        if params_str:
            result['params'] = _parse_python_params(params_str)
            if result['params'] and result['params'][0]['name'] in ('self', 'cls'):
                result['is_method'] = True

    return result


def _parse_python_params(params_str: str) -> List[Dict[str, Any]]:
    """comma-split respecting brackets, then parse each param."""
    parts: List[str] = []
    depth = 0
    current = ''
    for ch in params_str:
        if ch in '([{':
            depth += 1
            current += ch
        elif ch in ')]}':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    params = []
    for part in parts:
        if not part or part in ('/', '*') or part.strip() == '...':
            continue
        var_kw = part.startswith('**')
        var_pos = part.startswith('*') and not var_kw
        clean = part.lstrip('*')

        default = None
        if '=' in clean:
            name_ann, default = clean.split('=', 1)
            name_ann = name_ann.strip()
            default = default.strip()
        else:
            name_ann = clean

        ann = None
        if ':' in name_ann:
            name_part, ann = name_ann.split(':', 1)
            name = name_part.strip()
            ann = ann.strip()
        else:
            name = name_ann.strip()

        params.append({
            'name': name,
            'type': ann,
            'default': default,
            'var_positional': var_pos,
            'var_keyword': var_kw,
        })

    return params


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def _extract_imports(code: str) -> List[str]:
    """Extract import statements from code (AST first, regex fallback)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return re.findall(r'(?:^|\n)\s*((?:import|from)\s+[^\n]+)', code)

    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                part = f"import {alias.name}"
                if alias.asname:
                    part += f" as {alias.asname}"
                imports.append(part)
        elif isinstance(node, ast.ImportFrom):
            names_str = ', '.join(
                (f"{a.name} as {a.asname}" if a.asname else a.name)
                for a in node.names
            )
            imports.append(f"from {node.module or ''} import {names_str}")
    return imports


# ---------------------------------------------------------------------------
# Python AST code analyzer
# ---------------------------------------------------------------------------

class _PythonCodeAnalyzer(ast.NodeVisitor):
    """Python AST visitor — extract Code-Context View nodes and edges."""

    def __init__(self, api_name: str) -> None:
        self.api_name = api_name
        self.api_short = api_name.split('.')[-1] if '.' in api_name else api_name
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self._func_stack: List[str] = []
        self._counters: Dict[str, int] = {}

    def _uid(self, prefix: str) -> str:
        c = self._counters.get(prefix, 0)
        self._counters[prefix] = c + 1
        return prefix if c == 0 else f"{prefix}_{c}"

    def _add_node(self, nid: str, **kw) -> str:
        self.nodes.append({"id": nid, **{k: v for k, v in kw.items() if v is not None}})
        return nid

    def _add_edge(self, from_: str, to_: str, type_: str) -> None:
        self.edges.append({"from": from_, "to": to_, "type": type_})

    # ---- FunctionDef / AsyncFunctionDef ----

    def _visit_function(self, node: ast.FunctionDef) -> None:
        func_id = self._uid(f"code_function_{node.name}")
        ret_ann = None
        if node.returns:
            try:
                ret_ann = ast.unparse(node.returns)
            except Exception:
                pass
        self._add_node(func_id, type="code_function", name=node.name, return_annotation=ret_ann)

        for arg in node.args.args:
            ann = None
            if arg.annotation:
                try:
                    ann = ast.unparse(arg.annotation)
                except Exception:
                    pass
            param_id = self._uid(f"code_param_{arg.arg}")
            self._add_node(param_id, type="code_param", name=arg.arg, annotation=ann)
            self._add_edge(func_id, param_id, "has_param")

        # Detect bare-decorator API usage (e.g. @cache, @functools.cache)
        # generic_visit only traverses ast.Call decorators; bare refs need separate handling
        for decorator in node.decorator_list:
            is_bare = isinstance(decorator, (ast.Name, ast.Attribute))
            if is_bare:
                try:
                    dec_str = ast.unparse(decorator)
                except Exception:
                    dec_str = ''
                if (dec_str == self.api_short or
                        dec_str == self.api_name or
                        dec_str.endswith(f'.{self.api_short}')):
                    call_id = self._uid("code_api_call")
                    self._add_node(call_id, type="code_api_call",
                                   api=self.api_name, call_repr=f"@{dec_str}")
                    self._add_edge(func_id, call_id, "decorated_by")

        self._func_stack.append(func_id)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # ---- Call ----

    def _is_api_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id == self.api_short
        if isinstance(func, ast.Attribute):
            if func.attr == self.api_short:
                return True
            try:
                full = ast.unparse(func)
                return full == self.api_name or full.endswith(f".{self.api_short}")
            except Exception:
                pass
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_api_call(node):
            call_id = self._uid("code_api_call")
            try:
                call_repr = ast.unparse(node)[:200]
            except Exception:
                call_repr = self.api_short
            self._add_node(call_id, type="code_api_call", api=self.api_name, call_repr=call_repr)

            if self._func_stack:
                self._add_edge(self._func_stack[-1], call_id, "calls")

            for i, arg in enumerate(node.args):
                try:
                    val = ast.unparse(arg)[:80]
                except Exception:
                    val = f"arg_{i}"
                arg_id = self._uid(f"code_call_arg_pos{i}")
                self._add_node(arg_id, type="code_call_arg", position=i, value=val)
                self._add_edge(call_id, arg_id, "has_arg")

            for kwarg in node.keywords:
                try:
                    kval = ast.unparse(kwarg.value)[:80]
                except Exception:
                    kval = "..."
                kname = kwarg.arg or "**kwargs"
                kw_id = self._uid(f"code_call_arg_kw_{kname}")
                self._add_node(kw_id, type="code_call_arg", name=kname, value=kval, keyword=True)
                self._add_edge(call_id, kw_id, "has_kwarg")

        self.generic_visit(node)


def _analyze_python_code_structure(code: str, api_name: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyse Reference Code with Python AST; extract Code-Context View nodes and edges.

    Returns: (nodes, edges)
    """
    if not code or not code.strip():
        return [], []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Regex fallback
        nodes: List[Dict] = []
        api_short = api_name.split('.')[-1] if '.' in api_name else api_name
        for m in re.finditer(r'def\s+(\w+)\s*\(', code):
            nodes.append({"id": f"code_function_{m.group(1)}", "type": "code_function", "name": m.group(1)})
        for i, m in enumerate(re.finditer(rf'\b{re.escape(api_short)}\s*\(', code)):
            nodes.append({"id": f"code_api_call_{i}", "type": "code_api_call", "api": api_name})
        return nodes, []

    analyzer = _PythonCodeAnalyzer(api_name)
    analyzer.visit(tree)
    return analyzer.nodes, analyzer.edges


# ---------------------------------------------------------------------------
# Evolution sub-graph extractors (one per change_type)
# ---------------------------------------------------------------------------

def _extract_signature_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    signature: Signature evolution via old_signature vs signature comparison.
    """
    api_name = sample.get('name', '')
    old_sig = sample.get('old_signature', '')
    new_sig = sample.get('signature', '')
    code = sample.get('code', '')

    # Nothing to compare
    if not old_sig and not new_sig:
        return None

    imports = _extract_imports(code)
    old_parsed = parse_python_signature(old_sig) if old_sig else {}
    new_parsed = parse_python_signature(new_sig) if new_sig else {}

    # Compute diff
    old_params_map = {p['name']: p for p in old_parsed.get('params', []) if p['name'] not in ('self', 'cls')}
    new_params_map = {p['name']: p for p in new_parsed.get('params', []) if p['name'] not in ('self', 'cls')}
    old_names = set(old_params_map)
    new_names = set(new_params_map)

    changes: List[Dict] = []
    for n in old_names - new_names:
        changes.append({"kind": "param_removed", "name": n, "old_type": old_params_map[n].get('type')})
    for n in new_names - old_names:
        changes.append({"kind": "param_added", "name": n, "new_type": new_params_map[n].get('type')})
    for n in old_names & new_names:
        if old_params_map[n].get('type') != new_params_map[n].get('type'):
            changes.append({"kind": "param_type_changed", "name": n,
                            "old_type": old_params_map[n].get('type'),
                            "new_type": new_params_map[n].get('type')})

    old_ret = old_parsed.get('return_type')
    new_ret = new_parsed.get('return_type')
    if old_ret != new_ret:
        changes.append({"kind": "return_type_changed", "old": old_ret, "new": new_ret})

    nodes: List[Dict] = []
    edges: List[Dict] = []

    old_sig = _clean_sig(old_sig)
    new_sig = _clean_sig(new_sig)

    if old_sig:
        nodes.append({
            "id": "old_signature",
            "type": "old_signature",
            "api": api_name,
            "name": old_parsed.get('name') or api_name.split('.')[-1],
            "params": old_parsed.get('params', []),
            "return_type": old_parsed.get('return_type'),
            "role": "old",
        })
    if new_sig:
        nodes.append({
            "id": "new_signature",
            "type": "new_signature",
            "api": api_name,
            "name": new_parsed.get('name') or api_name.split('.')[-1],
            "params": new_parsed.get('params', []),
            "return_type": new_parsed.get('return_type'),
            "role": "new",
        })
    if old_sig and new_sig:
        edges.append({"from": "old_signature", "to": "new_signature", "type": "evolves"})

    if changes:
        nodes.append({
            "id": "sig_change",
            "type": "change",
            "api": api_name,
            "kind": "signature",
            "changes": changes,
        })
        if old_sig:
            edges.append({"from": "sig_change", "to": "old_signature", "type": "has_change"})
        if new_sig:
            edges.append({"from": "sig_change", "to": "new_signature", "type": "has_change"})

    return {
        "nodes": nodes,
        "edges": edges,
        "imports": imports,
        "changes": changes,
        "old_parsed": old_parsed,
        "new_parsed": new_parsed,
    }


def _extract_stabilized_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    stabilized: API from experimental/unstable to stable.
    """
    api_name = sample.get('name', '')
    from_version = sample.get('from_version', '')
    to_version = sample.get('to_version', '')
    signature = sample.get('signature', '')
    documentation = sample.get('documentation', '')
    code = sample.get('code', '')

    imports = _extract_imports(code)

    # stabilized signatures use "api.name(...)" format (not def-prefixed)
    # Extract parameter list via regex
    sig_clean = _clean_sig(signature.split(' — ')[0].split(' - ')[0]) if signature else ''
    sig_summary = ''
    if sig_clean:
        m = re.search(r'\(([^)]*)\)\s*$', sig_clean)
        if m:
            params_raw = m.group(1).strip()
            if params_raw:
                param_names = []
                for p in params_raw.split(','):
                    pn = p.split(':')[0].split('=')[0].strip().lstrip('*').strip()
                    if pn and pn not in ('self', 'cls', '...'):
                        param_names.append(pn)
                if param_names:
                    sig_summary = f"({', '.join(param_names[:5])})"

    nodes: List[Dict] = [
        {
            "id": "old",
            "type": "api",
            "api": api_name,
            "version": from_version,
            "status": "unstable/experimental",
            "signature_summary": sig_summary,
            "role": "old",
        },
        {
            "id": "new",
            "type": "api",
            "api": api_name,
            "version": to_version,
            "status": "stable",
            "signature_summary": sig_summary,
            "role": "new",
        },
    ]
    edges: List[Dict] = [
        {"from": "old", "to": "new", "type": "evolves"},
    ]

    return {"nodes": nodes, "edges": edges, "imports": imports}


def _extract_deprecated_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    deprecated: Old API deprecated, apply new replacement API.
    """
    api_name = sample.get('name', '')
    old_signature = sample.get('old_signature', '')
    new_signature = sample.get('signature', '')
    description = sample.get('description', '')
    documentation = sample.get('documentation', '')
    code = sample.get('code', '')

    imports = _extract_imports(code)

    # Prefer extracting replacement API name from description/documentation
    replacement = None
    for text in [description, documentation]:
        if text:
            m = re.search(
                r'(?:use|replaced\s+by|replaced\s+with|instead\s+use)\s+[`"\']?([\w][\w.]*)[`"\']?',
                text, re.IGNORECASE,
            )
            if m:
                replacement = m.group(1)
                break

    # Fallback: extract replacement API name from signature field
    # For deprecated type, signature is the replacement API call form
    if not replacement and new_signature:
        sig_clean_dep = new_signature.strip()
        # Handle import statements: "from X import Y" or "import X.Y.Z"
        m_imp = re.match(r'from\s+([\w.]+)\s+import\s+([\w]+)', sig_clean_dep)
        m_imp2 = re.match(r'import\s+([\w.]+)', sig_clean_dep)
        if m_imp:
            replacement = m_imp.group(1) + '.' + m_imp.group(2)
        elif m_imp2:
            replacement = m_imp2.group(1)
        else:
            # Normal function call: "funcname(...)" or "module.funcname(...)"
            m_func = re.match(r'^([\w][\w.]*(?:\.[\w]+)*)\s*[\(\s]', sig_clean_dep)
            if m_func:
                cand = m_func.group(1).strip('.')
                # Skip builtin generic functions (list, dict, etc.)
                if cand not in ('list', 'dict', 'tuple', 'set', 'str', 'int', 'float'):
                    replacement = cand

    nodes: List[Dict] = [
        {
            "id": "old",
            "type": "api",
            "api": api_name,
            "status": "deprecated",
            "role": "old",
        },
    ]
    edges: List[Dict] = []

    if replacement:
        nodes.append({
            "id": "new",
            "type": "api",
            "api": replacement,
            "status": "active",
            "role": "new",
        })
        edges.append({"from": "old", "to": "new", "type": "deprecated_via"})
    elif new_signature and new_signature != old_signature:
        # No explicit replacement name; use generalised new api node
        nodes.append({
            "id": "new",
            "type": "api",
            "api": api_name,
            "status": "new",
            "role": "new",
        })
        edges.append({"from": "old", "to": "new", "type": "recommends"})

    # Add structured signature nodes (old API params vs new/replacement API params)
    api_short_dep = api_name.split('.')[-1]
    if old_signature:
        old_parsed_dep = parse_python_signature(old_signature)
        if old_parsed_dep.get('params') is not None:
            nodes.append({
                "id": "old_signature",
                "type": "old_signature",
                "api": api_name,
                "name": old_parsed_dep.get('name') or api_short_dep,
                "params": old_parsed_dep.get('params', []),
                "return_type": old_parsed_dep.get('return_type'),
                "role": "old",
            })
            edges.append({"from": "old", "to": "old_signature", "type": "has_change"})

    if new_signature and new_signature != old_signature:
        # Try parsing signature field (may be "funcname(params)" format)
        sig_for_new = new_signature.split(' — ')[0].strip()
        # Strip # comments (e.g. "func()  # or other()")
        sig_for_new = re.sub(r'\s*#.*$', '', sig_for_new).strip()
        # Truncate to first matching paren close
        depth, cut = 0, len(sig_for_new)
        for i, ch in enumerate(sig_for_new):
            if ch == '(': depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0: cut = i + 1; break
        sig_for_new = sig_for_new[:cut].strip()
        if not sig_for_new.startswith(('def ', 'async def ', 'from ', 'import ')):
            sig_for_new = 'def ' + sig_for_new
        new_parsed_dep = parse_python_signature(sig_for_new)
        if new_parsed_dep.get('params') is not None:
            rep_name = (replacement or '').split('.')[-1] or api_short_dep
            nodes.append({
                "id": "new_signature",
                "type": "new_signature",
                "api": replacement or api_name,
                "name": new_parsed_dep.get('name') or rep_name,
                "params": new_parsed_dep.get('params', []),
                "return_type": new_parsed_dep.get('return_type'),
                "role": "new",
            })
            edges.append({"from": "new" if replacement else "old",
                          "to": "new_signature", "type": "defines_current"})

    return {"nodes": nodes, "edges": edges, "imports": imports, "replacement": replacement}


def _extract_implicit_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    implicit: Signature unchanged but behaviour/semantics changed.
    """
    api_name = sample.get('name', '')
    signature = sample.get('signature', '')
    old_signature = sample.get('old_signature', '')
    description = sample.get('description', '')
    documentation = sample.get('documentation', '')
    from_version = sample.get('from_version', '')
    to_version = sample.get('to_version', '')
    code = sample.get('code', '')

    imports = _extract_imports(code)

    # implicit signature format: "funcname(params) — behavior_description"
    # Before dash: call form; after dash: semantic description distinguishing old/new
    def _extract_behavior_desc(sig_text: str) -> str:
        """Extract behaviour description after em-dash (short semantic label)."""
        if not sig_text:
            return ''
        parts = sig_text.split(' — ')
        if len(parts) > 1:
            return parts[1].strip()[:120]
        return ''

    new_behavior_desc = _extract_behavior_desc(signature)
    old_behavior_desc = _extract_behavior_desc(old_signature)

    # Parse signature params for new_signature/old_signature nodes (implicit: signature unchanged)
    sig_raw = signature.split(' — ')[0].split(' - ')[0].strip() if signature else ''
    sig_clean = _clean_sig(sig_raw)
    if sig_clean and not sig_clean.startswith(('def ', 'async def ')):
        sig_for_parse = 'def ' + sig_clean
    else:
        sig_for_parse = sig_clean
    sig_parsed = parse_python_signature(sig_for_parse) if sig_for_parse else {}

    nodes: List[Dict] = [
        {
            "id": "old",
            "type": "api",
            "api": api_name,
            "version": from_version,
            "status": "behavior",
            "behavior_desc": old_behavior_desc,
            "role": "old",
        },
        {
            "id": "new",
            "type": "api",
            "api": api_name,
            "version": to_version,
            "status": "behavior",
            "behavior_desc": new_behavior_desc,
            "role": "new",
        },
    ]
    edges: List[Dict] = [
        {"from": "old", "to": "new", "type": "evolves"},
    ]

    # Add signature node for implicit type (signature unchanged, but provides structured interface info)
    if sig_parsed.get('params') is not None:
        api_short = api_name.split('.')[-1]
        nodes.append({
            "id": "new_signature",
            "type": "new_signature",
            "api": api_name,
            "name": sig_parsed.get('name') or api_short,
            "params": sig_parsed.get('params', []),
            "return_type": sig_parsed.get('return_type'),
            "role": "new",
        })
        edges.append({"from": "new", "to": "new_signature", "type": "defines_current"})

    return {"nodes": nodes, "edges": edges, "imports": imports, "sig_parsed": sig_parsed}


def extract_python_api_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Dispatch to change_type-specific evolution extractor.
    """
    ct = sample.get('change_type', '')
    if ct == 'stabilized':
        return _extract_stabilized_evolution(sample)
    elif ct == 'signature':
        return _extract_signature_evolution(sample)
    elif ct == 'deprecated':
        return _extract_deprecated_evolution(sample)
    elif ct == 'implicit':
        return _extract_implicit_evolution(sample)
    else:
        return _extract_signature_evolution(sample)


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------

def _clean_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None / empty-string / empty-list values."""
    return {k: v for k, v in node.items() if v is not None and v != '' and v != []}


def build_graph_entry(idx: int, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build heterogeneous API evolution graph for a single PyEvo sample.

    Output format (compatible with RustEvo):
    {
        "id":        int,
        "signature": str,
        "nodes":     [ {"id": str, "type": str, ...}, ... ],
        "edges":     [ {"from": str, "to": str, "type": str}, ... ],
        "api":       { "name": str, "module": str, "change_type": str,
                       "from_version": str, "to_version": str },
        "code":      str,
    }
    """
    api_name      = sample.get('name', '')
    api_module    = sample.get('module', '')
    change_type   = sample.get('change_type', 'signature')
    from_version  = sample.get('from_version', '')
    to_version    = sample.get('to_version', '')
    signature     = sample.get('signature', '')
    code          = sample.get('code', '')
    func_sig_raw  = sample.get('function_signature', '')

    # ===== 1. Analyze user code structure (ast) =====
    code_nodes, code_edges = _analyze_python_code_structure(code, api_name)

    # ===== 2. Parse function_signature =====
    # Guard: function_signature can sometimes be a multiline code block (data noise).
    # Only parse if it looks like a single-line def signature.
    if func_sig_raw and '\n' in func_sig_raw:
        first_line = func_sig_raw.split('\n')[0].strip()
        if first_line.startswith(('def ', 'async def ')):
            func_sig_raw = first_line   # use only the def line
        else:
            func_sig_raw = ''           # not a usable signature
    func_sig_parsed = parse_python_signature(func_sig_raw) if func_sig_raw else {}

    # ===== 3. Extract API evolution sub-graph =====
    evolution_data = extract_python_api_evolution(sample)

    # ===== 4. Assemble graph =====
    all_nodes: List[Dict] = []
    all_edges: List[Dict] = []

    # --- 4a. api_updated anchor ---
    API_ID = "api"
    all_nodes.append({
        "id": API_ID,
        "type": "api_updated",
        "name": api_name,
        "module": api_module,
        "change_type": change_type,
        "from_version": from_version,
        "to_version": to_version,
    })

    # --- 4b. Code structure nodes (prefix with "code_" if not already) ---
    code_id_map: Dict[str, str] = {}
    for node in code_nodes:
        oid = node.get('id', '')
        nid = oid if oid.startswith('code_') else f"code_{oid}"
        code_id_map[oid] = nid
        n = dict(node)
        n['id'] = nid
        all_nodes.append(n)

    for edge in code_edges:
        all_edges.append({
            "from": code_id_map.get(edge['from'], f"code_{edge['from']}"),
            "to":   code_id_map.get(edge['to'],   f"code_{edge['to']}"),
            "type": edge['type'],
        })

    # Connect first code_function → api via implements_query
    first_func = next((n for n in all_nodes if n.get('type') == 'code_function'), None)
    if first_func:
        all_edges.append({"from": first_func['id'], "to": API_ID, "type": "implements_query"})

    # Connect each code_api_call → api via calls (bridge edge)
    for n in all_nodes:
        if n.get('type') == 'code_api_call':
            all_edges.append({"from": n['id'], "to": API_ID, "type": "calls"})

    # --- 4c. Evolution nodes (prefix with "evo_" if not already) ---
    if evolution_data:
        evo_id_map: Dict[str, str] = {}
        for node in evolution_data.get('nodes', []):
            oid = node.get('id', '')
            nid = oid if oid.startswith('evo_') or oid in ('old_signature', 'new_signature') else f"evo_{oid}"
            evo_id_map[oid] = nid
            n = dict(node)
            n['id'] = nid
            all_nodes.append(n)

        for edge in evolution_data.get('edges', []):
            all_edges.append({
                "from": evo_id_map.get(edge['from'], f"evo_{edge['from']}"),
                "to":   evo_id_map.get(edge['to'],   f"evo_{edge['to']}"),
                "type": edge['type'],
            })

        # Connect evolution nodes to api anchor
        type_to_anchor_edge: Dict[str, str] = {
            'new_signature': 'defines_current',
            'old_signature': 'defines_previous',
        }
        for n in all_nodes:
            ntype_a = n.get('type', '')
            role_a = n.get('role', '')
            if ntype_a in type_to_anchor_edge:
                edge_type = type_to_anchor_edge[ntype_a]
            elif ntype_a == 'api' and role_a == 'new':
                edge_type = 'defines_current'
            elif ntype_a == 'api' and role_a == 'old':
                edge_type = 'defines_previous'
            else:
                continue
            all_edges.append({"from": API_ID, "to": n['id'], "type": edge_type})

    # --- 4d. func_param / func_return nodes from parsed function_signature ---
    func_param_ids: List[str] = []
    if func_sig_parsed:
        for param in func_sig_parsed.get('params', []):
            pname = param.get('name', '')
            if pname in ('self', 'cls'):
                continue
            param_id = f"func_param_{pname}"
            _ann = (param.get('type') or '') or None
            _def = (param.get('default') or '') or None
            all_nodes.append({
                "id": param_id,
                "type": "func_param",
                "name": pname,
                "annotation": _ann,
                "default": _def,
            })
            func_param_ids.append(param_id)
            all_edges.append({"from": API_ID, "to": param_id, "type": "constrains_param"})

        ret_type = func_sig_parsed.get('return_type')
        if ret_type:
            all_nodes.append({"id": "func_return", "type": "func_return", "annotation": ret_type})
            all_edges.append({"from": API_ID, "to": "func_return", "type": "constrains_return"})

        # constrains_arg: func_param → matching code_param (by name)
        code_params = [n for n in all_nodes if n.get('type') == 'code_param']
        for fp_id in func_param_ids:
            fp_node = next((n for n in all_nodes if n['id'] == fp_id), None)
            if not fp_node:
                continue
            fp_name = fp_node.get('name', '')
            for cp in code_params:
                if cp.get('name') == fp_name:
                    all_edges.append({"from": fp_id, "to": cp['id'], "type": "constrains_arg"})
                    break
            # Also try to match code_call_arg by kwarg name
            for n in all_nodes:
                if n.get('type') == 'code_call_arg' and n.get('name') == fp_name:
                    all_edges.append({"from": fp_id, "to": n['id'], "type": "constrains_arg"})

        # new_signature → func_param constraint edges
        new_sig_node = next((n for n in all_nodes if n.get('type') == 'new_signature'), None)
        if new_sig_node:
            for sp in new_sig_node.get('params', []):
                sp_name = sp.get('name', '')
                if sp_name in ('self', 'cls', '...'):
                    continue
                # Find matching func_param node by name and connect
                for n in all_nodes:
                    if n.get('type') == 'func_param' and n.get('name') == sp_name:
                        all_edges.append({"from": new_sig_node['id'], "to": n['id'],
                                          "type": "sig_param_matches"})

    # --- 4e. API-call-centric slicing (3-hop BFS from code_api_call / api_anchor) ---
    api_call_ids = {n['id'] for n in all_nodes if n.get('type') == 'code_api_call'}
    anchor_ids = api_call_ids | {API_ID}

    if anchor_ids:
        adj: Dict[str, set] = {}
        for edge in all_edges:
            adj.setdefault(edge['from'], set()).add(edge['to'])
            adj.setdefault(edge['to'],   set()).add(edge['from'])

        keep_ids: set = set(anchor_ids)
        frontier = set(anchor_ids)
        for _ in range(3):
            nxt: set = set()
            for nid in frontier:
                for nb in adj.get(nid, set()):
                    if nb not in keep_ids:
                        keep_ids.add(nb)
                        nxt.add(nb)
            frontier = nxt

        # Always keep evolution, func_* and api_updated nodes
        _ALWAYS_KEEP_PREFIXES = ('evo_', 'func_')
        _ALWAYS_KEEP_TYPES = ('old_signature', 'new_signature', 'api_updated')
        for n in all_nodes:
            nt = n.get('type', '')
            role_k = n.get('role', '')
            if (any(nt.startswith(p) for p in _ALWAYS_KEEP_PREFIXES)
                    or nt in _ALWAYS_KEEP_TYPES
                    or (nt == 'api' and role_k in ('old', 'new'))):
                keep_ids.add(n['id'])

        all_nodes = [n for n in all_nodes if n['id'] in keep_ids]
        all_node_ids = {n['id'] for n in all_nodes}
        all_edges = [e for e in all_edges if e['from'] in all_node_ids and e['to'] in all_node_ids]

    # --- 4f. Clean, deduplicate, sort ---
    all_nodes = [_clean_node(n) for n in all_nodes]

    seen_edge_keys: set = set()
    dedup_edges: List[Dict] = []
    for e in all_edges:
        key = (e['from'], e['to'], e['type'])
        if key not in seen_edge_keys:
            seen_edge_keys.add(key)
            dedup_edges.append(e)

    TYPE_ORDER = {
        'api_updated': 0,
        'code_function': 1,
        'code_param': 2,
        'code_return': 3,
        'code_api_call': 4,
        'code_call_arg': 5,
        'func_param': 6,
        'func_return': 7,
    }

    def _node_sort_key(n: Dict) -> Tuple:
        nt = n.get('type', 'z')
        role_s = n.get('role', '')
        base = TYPE_ORDER.get(nt, 8 if (nt.startswith('evo_') or (nt == 'api' and role_s in ('old', 'new'))) else 10)
        return (base, n.get('id', ''))

    all_nodes.sort(key=_node_sort_key)
    dedup_edges.sort(key=lambda e: (e['from'], e['to'], e['type']))

    return {
        "id":        idx,
        "signature": func_sig_raw or signature,
        "nodes":     all_nodes,
        "edges":     dedup_edges,
        "api": {
            "name":         api_name,
            "module":       api_module,
            "change_type":  change_type,
            "from_version": from_version,
            "to_version":   to_version,
        },
        "code": code,
    }


# ---------------------------------------------------------------------------
# Dataset-level builder
# ---------------------------------------------------------------------------

def build_and_save_graphs(dataset, output_dir: str):
    """
    Build graphs for all PyEvo samples and save as JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "pyevo_graphs.json"

    print(f"\nBuilding PyEvo graphs: {len(dataset)} samples")
    print(f"Output dir: {output_path}")

    graphs: List[Optional[Dict]] = []
    stats: Dict[str, Any] = {
        'total': len(dataset),
        'built': 0,
        'failed': 0,
        'total_nodes': 0,
        'total_edges': 0,
        'node_types': {},
        'edge_types': {},
        'by_change_type': {},
    }

    t0 = perf_counter()
    for idx in tqdm(range(len(dataset)), desc="Building PyEvo graphs"):
        sample = dataset[idx]
        ct = sample.get('change_type', 'unknown')
        stats['by_change_type'].setdefault(ct, {'built': 0, 'failed': 0})

        try:
            graph = build_graph_entry(idx, sample)
            graphs.append(graph)
            stats['built'] += 1
            stats['by_change_type'][ct]['built'] += 1

            if graph:
                stats['total_nodes'] += len(graph['nodes'])
                stats['total_edges'] += len(graph['edges'])
                for node in graph['nodes']:
                    ntype = node.get('type', 'unknown')
                    stats['node_types'][ntype] = stats['node_types'].get(ntype, 0) + 1
                for edge in graph['edges']:
                    etype = edge.get('type', 'unknown')
                    stats['edge_types'][etype] = stats['edge_types'].get(etype, 0) + 1

            if (idx + 1) % 100 == 0 and graph:
                print(f"  [{idx + 1}/{len(dataset)}] nodes={len(graph['nodes'])}, "
                      f"edges={len(graph['edges'])}, change_type={ct}")
        except Exception as exc:
            print(f"\n  [Error] Sample {idx} ({ct}): {exc}")
            graphs.append(None)
            stats['failed'] += 1
            stats['by_change_type'][ct]['failed'] += 1

    elapsed = perf_counter() - t0
    stats['elapsed_seconds'] = round(elapsed, 2)
    stats['avg_nodes'] = round(stats['total_nodes'] / max(stats['built'], 1), 1)
    stats['avg_edges'] = round(stats['total_edges'] / max(stats['built'], 1), 1)

    # Save graphs
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graphs, f, ensure_ascii=False, indent=2)
    print(f"\nGraphs saved: {output_file}")

    # Save stats
    stats_file = output_path / "pyevo_graphs_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done: built={stats['built']}, failed={stats['failed']}, time={elapsed:.1f}s")
    print(f"Total nodes: {stats['total_nodes']} (avg {stats['avg_nodes']}/sample)")
    print(f"Total edges: {stats['total_edges']} (avg {stats['avg_edges']}/sample)")
    print("\nNode type distribution:")
    for ntype, cnt in sorted(stats['node_types'].items(), key=lambda x: -x[1]):
        print(f"  {ntype:30s}: {cnt}")
    print("\nEdge type distribution:")
    for etype, cnt in sorted(stats['edge_types'].items(), key=lambda x: -x[1]):
        print(f"  {etype:30s}: {cnt}")
    print("\nBy change_type:")
    for ct_k, ct_v in stats['by_change_type'].items():
        print(f"  {ct_k:15s}: built={ct_v['built']}, failed={ct_v['failed']}")
    print(f"{'='*60}")

    return graphs, stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build PyEvo heterogeneous API evolution graphs")
    parser.add_argument(
        '--output_dir', type=str, default='./data/pyevo_graphs',
        help='Graph output directory',
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='PyEvo dataset root (defaults to util/globals.DATA_DIR)',
    )
    parser.add_argument(
        '--size', type=int, default=None,
        help='Limit number of samples (for debugging)',
    )
    parser.add_argument(
        '--model_name', type=str, default='Qwen2.5-7B-Instruct',
        help='Model name for PyEvoDataset formatting',
    )
    args = parser.parse_args()

    data_dir = args.data_dir or str(DATA_DIR)
    print(f"Loading PyEvoDataset from: {data_dir}")
    dataset = PyEvoDataset(data_dir, model_name=args.model_name, size=args.size)
    print(f"Dataset size: {len(dataset)}")

    build_and_save_graphs(dataset, args.output_dir)


if __name__ == '__main__':
    main()

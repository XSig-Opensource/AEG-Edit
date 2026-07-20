"""
build_rustevo_graphs.py — Heterogeneous API Evolution Graph Builder for RustEvo

Constructs a heterogeneous graph G = (V_api ∪ V_code, E) for each sample,
covering both the API evolution history (Evolution-Aware View) and the
surrounding code context (Code-Context View). A 3-hop BFS prunes nodes
irrelevant to the target API call site.

Usage: python scripts/build_rustevo_graphs.py --output_dir ./data/rustevo_graphs
"""

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dsets import RustEvoDataset
from util.globals import DATA_DIR


def parse_function_signature(sig: str) -> Dict[str, Any]:
    """Parse a Rust function_signature into structured components.

    Returns dict with keys: name, visibility, generics, params, return_type.
    """
    result = {
        'name': '',
        'visibility': None,
        'generics': [],
        'params': [],
        'return_type': None
    }
    
    if not sig:
        return result
    
    # Extract visibility modifier
    if sig.strip().startswith('pub '):
        result['visibility'] = 'pub'
    elif 'pub(crate)' in sig:
        result['visibility'] = 'pub(crate)'
    
    # Extract function name
    name_match = re.search(r'\bfn\s+(\w+)', sig)
    if name_match:
        result['name'] = name_match.group(1)
    
    # Extract generics: fn foo<T, U: Clone>
    generic_match = re.search(r'\bfn\s+\w+\s*<([^>]+)>', sig)
    if generic_match:
        generics_str = generic_match.group(1)
        result['generics'] = [g.strip() for g in _split_generics(generics_str)]
    
    # Extract parameter list (handles nested < >)
    params_str = _extract_params(sig)
    if params_str:
        for param in _split_params(params_str):
            param = param.strip()
            if param and ':' in param:
                parts = param.split(':', 1)
                result['params'].append({
                    'name': parts[0].strip(),
                    'type': parts[1].strip()
                })
    
    # Extract return type
    ret_match = re.search(r'->\s*(.+?)(?:\s*(?:where|\{)|$)', sig, re.DOTALL)
    if ret_match:
        ret = ret_match.group(1).strip()
        ret = ret.split('\n')[0].strip()  # Keep first line only
        # Strip trailing braces
        brace_idx = ret.find('{')
        if brace_idx != -1:
            ret = ret[:brace_idx].strip()
        result['return_type'] = ret
    
    return result


def _parse_impl_block(sig: str) -> Optional[Dict[str, Any]]:
    """Parse an impl block with complex generics.

    Returns dict with keys: generics, impl_trait, impl_for, name.
    """
    result = {'generics': [], 'impl_trait': '', 'impl_for': '', 'name': ''}
    
    # Strip unsafe/const prefix, locate content after `impl`
    match = re.match(r'(?:unsafe\s+)?impl\s*(?:const\s+)?', sig)
    if not match:
        return None
    
    rest = sig[match.end():]
    
    # 1. Extract impl generics <T, U>
    if rest.startswith('<'):
        generics_str = _extract_balanced(rest, '<', '>')
        if generics_str:
            result['generics'] = [g.strip() for g in _split_generics(generics_str)]
            rest = rest[len(generics_str) + 2:].strip()  # +2 for < and >
    
    # 2. Locate " for " keyword (space-bounded to avoid matching 'Format')
    for_match = re.search(r'\s+for\s+', rest)
    if not for_match:
        return None
    
    # 3. Trait part (before `for`)
    trait_part = rest[:for_match.start()].strip()
    result['impl_trait'] = trait_part
    
    # 4. Type part (after `for`)
    type_part = rest[for_match.end():].strip()
    # Strip where clauses, newlines, braces
    where_idx = type_part.find('\n')
    if where_idx != -1:
        type_part = type_part[:where_idx].strip()
    where_match = re.search(r'\bwhere\b', type_part)
    if where_match:
        type_part = type_part[:where_match.start()].strip()
    # Strip braces and trailing content
    brace_idx = type_part.find('{')
    if brace_idx != -1:
        type_part = type_part[:brace_idx].strip()
    result['impl_for'] = type_part
    
    # 5. Extract short name from trait name
    trait_name = trait_part
    # Strip generic parameters from trait name
    if '<' in trait_name:
        trait_name = trait_name[:trait_name.index('<')]
    if '::' in trait_name:
        trait_name = trait_name.split('::')[-1]
    result['name'] = trait_name
    
    return result


def _extract_balanced(s: str, open_char: str, close_char: str) -> Optional[str]:
    """Extract content within balanced delimiters (e.g. '<T, Option<U>>' -> 'T, Option<U>')."""

    if not s or s[0] != open_char:
        return None
    
    depth = 0
    for i, c in enumerate(s):
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return s[1:i]  # Exclude outer delimiters
    return None


def parse_api_signature(sig: str) -> Dict[str, Any]:
    """Parse an API signature into a structured dict.

    Supports fn, impl, struct, enum, const, and type_alias signatures.
    Returns dict with keys: kind, name, visibility, is_const, is_unsafe,
    generics, self_kind, params, return_type, where, impl_trait, impl_for.
    """
    result = {
        'kind': 'fn',
        'name': '',
        'visibility': None,
        'is_const': False,
        'is_unsafe': 'unsafe ' in sig,
        'generics': [],
        'self_kind': None,
        'params': [],
        'return_type': None,
        'where': None,
    }
    
    if not sig:
        return result
    
    # Extract visibility modifier
    vis_match = re.match(r'^(pub(?:\([^)]+\))?|pub\(crate\)|pub\(super\))\s+', sig)
    if vis_match:
        result['visibility'] = vis_match.group(1)
    
    # =========================================================================
    # Detect signature kind
    # =========================================================================
    
    # 1. struct
    struct_match = re.search(r'\bstruct\s+(\w+)', sig)
    if struct_match:
        result['kind'] = 'struct'
        result['name'] = struct_match.group(1)
        # Extract generics
        generic_match = re.search(r'\bstruct\s+\w+\s*<([^>]+)>', sig)
        if generic_match:
            result['generics'] = [g.strip() for g in _split_generics(generic_match.group(1))]
        return result
    
    # 2. enum
    enum_match = re.search(r'\benum\s+(\w+)', sig)
    if enum_match:
        result['kind'] = 'enum'
        result['name'] = enum_match.group(1)
        return result
    
    # 3. const
    const_match = re.search(r'\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*:', sig)
    if const_match and 'fn' not in sig:
        result['kind'] = 'const'
        result['name'] = const_match.group(1)
        # Extract type
        type_match = re.search(r'\bconst\s+\w+\s*:\s*([^=]+)', sig)
        if type_match:
            ret_type = type_match.group(1).strip()
            # Strip braces
            brace_idx = ret_type.find('{')
            if brace_idx != -1:
                ret_type = ret_type[:brace_idx].strip()
            result['return_type'] = ret_type
        return result
    
    # 4. type_alias
    type_match = re.search(r'\btype\s+(\w+)', sig)
    if type_match:
        result['kind'] = 'type_alias'
        result['name'] = type_match.group(1)
        # Extract generics
        generic_match = re.search(r'\btype\s+\w+\s*<([^>]+)>', sig)
        if generic_match:
            result['generics'] = [g.strip() for g in _split_generics(generic_match.group(1))]
        # Extract target type
        alias_match = re.search(r'=\s*([^;{]+)', sig)
        if alias_match:
            result['return_type'] = alias_match.group(1).strip()
        return result
    
    # 5. impl (supports unsafe impl, impl const, etc.)
    # Uses custom parser for complex generics
    if re.match(r'(?:unsafe\s+)?impl\b', sig):
        impl_info = _parse_impl_block(sig)
        if impl_info and impl_info.get('impl_trait') and impl_info.get('impl_for'):
            result['kind'] = 'impl'
            result['is_unsafe'] = sig.strip().startswith('unsafe')
            result['is_const'] = 'impl const' in sig or ' const ' in sig
            result.update(impl_info)
            return result
    
    # 6. fn
    result['is_const'] = 'const fn' in sig
    
    # Extract function name
    name_match = re.search(r'\bfn\s+(\w+)', sig)
    if name_match:
        result['name'] = name_match.group(1)
    
    # Extract generics
    generics_str = _extract_generics_after_fn(sig)
    if generics_str:
        result['generics'] = [g.strip() for g in _split_generics(generics_str)]
    
    # Extract params and detect self variants
    params_str = _extract_params(sig)
    if params_str:
        for param in _split_params(params_str):
            param = param.strip()
            if not param:
                continue
            
            # Check self variants
            if param in ('self', '&self', '&mut self'):
                result['self_kind'] = param
            elif re.match(r"&'\w+\s+(mut\s+)?self", param):
                result['self_kind'] = param
            elif param.startswith('self:'):
                result['self_kind'] = param
            elif ':' in param:
                parts = param.split(':', 1)
                result['params'].append({
                    'name': parts[0].strip(),
                    'type': parts[1].strip()
                })
    
    # Extract where clause first
    where_clause_raw = None
    where_match = re.search(r'\bwhere\s+(.+?)$', sig, re.DOTALL)
    if where_match:
        where_clause_raw = where_match.group(1).strip()
        where_clause_raw = ' '.join(where_clause_raw.split())
        result['where'] = _parse_where_clause(where_clause_raw)
        # Remove where clause to correctly extract return_type
        sig = sig[:where_match.start()].strip()
    
    # Extract return type (after removing where clause)
    ret_match = re.search(r'->\s*(.+?)$', sig, re.DOTALL)
    if ret_match:
        ret = ret_match.group(1).strip()
        ret = ret.split('\n')[0].strip()
        # Strip braces
        brace_idx = ret.find('{')
        if brace_idx != -1:
            ret = ret[:brace_idx].strip()
        result['return_type'] = ret
    
    return result


def _parse_where_clause(where_str: str) -> List[Dict[str, Any]]:
    """Parse a where clause into structured format.

    Returns list of dicts with keys: type_param, bounds.
    """
    if not where_str:
        return []
    
    result = []
    # Split by commas, handling nested brackets
    clauses = _split_where_clauses(where_str)
    
    for clause in clauses:
        clause = clause.strip()
        if not clause or ':' not in clause:
            continue
        
        # Split type parameter and bounds
        parts = clause.split(':', 1)
        if len(parts) != 2:
            continue
        
        type_param = parts[0].strip()
        bounds_str = parts[1].strip().rstrip(',')  # Strip trailing comma
        
        # Split multiple bounds (connected by +)
        bounds = [b.strip().rstrip(',') for b in bounds_str.split('+') if b.strip()]
        
        result.append({
            'type_param': type_param,
            'bounds': bounds
        })
    
    return result


def _split_where_clauses(where_str: str) -> List[str]:
    """Split where clauses handling nested parens and angle brackets."""
    result = []
    current = []
    depth_paren = 0
    depth_angle = 0
    
    i = 0
    while i < len(where_str):
        char = where_str[i]
        
        if char == '(':
            depth_paren += 1
            current.append(char)
        elif char == ')':
            depth_paren -= 1
            current.append(char)
        elif char == '<' and (i == 0 or where_str[i-1] != '-'):
            # Only count as angle bracket if not part of ->
            depth_angle += 1
            current.append(char)
        elif char == '>' and (i == 0 or where_str[i-1] != '-'):
            # Only count as angle bracket if not part of ->
            depth_angle -= 1
            current.append(char)
        elif char == ',' and depth_paren == 0 and depth_angle == 0:
            # Check if comma is followed by a colon (confirming a new clause)
            # e.g. "F: FnMut(&T) -> K, K: Ord" 
            # The first comma is followed by "K:", so it is a separator
            rest = where_str[i+1:].lstrip()
            # Find if next colon has only identifier chars before it
            next_colon_pos = rest.find(':')
            if next_colon_pos > 0:
                # Verify only identifier chars before colon
                before_colon = rest[:next_colon_pos].strip()
                if before_colon and before_colon.replace("'", "").isidentifier():
                    # New where clause detected; split here
                    if current:
                        result.append(''.join(current))
                        current = []
                    # Note: comma itself is consumed, not appended
                else:
                    # Comma is internal to a bound (e.g. function signature)
                    current.append(char)
            else:
                current.append(char)
        else:
            current.append(char)
        
        i += 1
    
    if current:
        result.append(''.join(current))
    
    return result


def _extract_params(sig: str) -> Optional[str]:
    """Extract the parameter string inside parentheses, handling nesting."""
    # Locate first '('
    start = sig.find('(')
    if start == -1:
        return None
    
    depth = 0
    end = start
    for i, c in enumerate(sig[start:], start):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    
    return sig[start + 1:end]


def _split_params(params_str: str) -> List[str]:
    """Split parameters handling nested < > and ( )."""
    result = []
    current = []
    depth = 0
    
    for c in params_str:
        if c in '<(':
            depth += 1
            current.append(c)
        elif c in '>)':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            result.append(''.join(current))
            current = []
        else:
            current.append(c)
    
    if current:
        result.append(''.join(current))
    
    return result


def _split_generics(generics_str: str) -> List[str]:
    """Split generic parameters handling nesting."""
    return _split_params(generics_str)

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate a list while preserving insertion order."""
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _slice_api_call_subgraph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    api_call_ids: List[str],
    hops: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Graph Pruning: BFS from API call nodes, retaining nodes within `hops` hops."""
    if not api_call_ids:
        return nodes, edges

    node_map = {n.get('id'): n for n in nodes if n.get('id') is not None}
    adj = {}
    for e in edges:
        src = e.get('from')
        dst = e.get('to')
        if src in node_map and dst in node_map:
            adj.setdefault(src, set()).add(dst)
            adj.setdefault(dst, set()).add(src)

    keep = set()
    frontier = set(api_call_ids)
    depth = 0
    while frontier and depth <= hops:
        keep.update(frontier)
        next_frontier = set()
        for nid in frontier:
            for nb in adj.get(nid, set()):
                if nb not in keep:
                    next_frontier.add(nb)
        frontier = next_frontier
        depth += 1

    # Always keep critical node types (V_API anchor, V_code essentials)
    always_keep_types = {'api_updated', 'code_function', 'func_param', 'func_return'}
    for n in nodes:
        nid = n.get('id')
        if nid is None:
            continue
        if n.get('type') in always_keep_types or nid == 'return_type':
            keep.add(nid)

    kept_nodes = [n for n in nodes if n.get('id') in keep]
    kept_edges = [e for e in edges if e.get('from') in keep and e.get('to') in keep]
    return kept_nodes, kept_edges


def _extract_generics_after_fn(sig: str) -> Optional[str]:
    """Extract generic parameters after fn name, handling nested <>."""
    # Locate first '<' after 'fn name'
    fn_match = re.search(r'\bfn\s+\w+\s*<', sig)
    if not fn_match:
        return None
    
    start = fn_match.end() - 1  # Points to '<'
    depth = 0
    end = start
    
    for i, c in enumerate(sig[start:], start):
        if c == '<':
            depth += 1
        elif c == '>':
            depth -= 1
            if depth == 0:
                end = i
                break
    
    if depth == 0 and end > start:
        return sig[start + 1:end]
    return None


# =============================================================================
# Node filtering (Graph Pruning helper)
# =============================================================================

# Node quality thresholds
MIN_NODE_TEXT_LENGTH = 3  # Minimum meaningful text length
USELESS_NODE_KEYWORDS = {'unknown', 'placeholder', 'none', 'empty', 'null'}
REQUIRED_FIELDS_MAP = {
    'type_dependencies': ['kind', 'name'],      # At least kind or name required
    'api_evolution': ['kind', 'type', 'name'],   # Evolution nodes need clear identity
}


def is_valid_node(node: Dict[str, Any], node_category: str = 'generic') -> bool:
    """Check whether a graph node passes quality filters.

    Criteria: must have id, required fields per category,
    meaningful text content, and purposeful placeholder nodes.
    """
    # 1. Must have id
    if not node.get('id'):
        return False
    
    # 2. Check required fields by node category
    if node_category in REQUIRED_FIELDS_MAP:
        required_fields = REQUIRED_FIELDS_MAP[node_category]
        # At least one required field must be non-empty
        if not any(node.get(field) for field in required_fields):
            return False
    
    # 3. Check key text fields for meaningful content
    # Placeholder nodes handled separately
    text_fields = ['value', 'name', 'kind', 'type']
    has_meaningful_text = False
    
    for field in text_fields:
        value = node.get(field, '')
        if value and isinstance(value, str):
            value_lower = value.lower().strip()
            # Check length and keywords
            if len(value_lower) >= MIN_NODE_TEXT_LENGTH:
                if value_lower not in USELESS_NODE_KEYWORDS:
                    has_meaningful_text = True
                    break
    
    # 4. Special node type checks
    node_type = node.get('type', '').lower()
    
    # Placeholder nodes: special handling
    if 'placeholder' in node_type:
        # Keep if it has a clear purpose (e.g. cross-graph bridge)
        purpose = node.get('purpose', '')
        connects_to = node.get('connects_to', '')
        # purpose/connects_to must have meaningful length
        if purpose and isinstance(purpose, str) and len(purpose.strip()) >= MIN_NODE_TEXT_LENGTH:
            return True  # Has clear purpose
        if connects_to and isinstance(connects_to, str) and len(str(connects_to).strip()) >= MIN_NODE_TEXT_LENGTH:
            return True  # Has connect target
        return False  # Purposeless placeholder
    
    # Non-placeholder nodes need meaningful text
    if not has_meaningful_text:
        return False
    
    # api_evolution nodes need explicit change info
    if node_category == 'api_evolution':
        if node_type in ['change', 'transition', 'signature']:
            # Must have kind or changes field
            if not node.get('kind') and not node.get('changes'):
                return False
    
    return True


def filter_graph_nodes(graph: Dict[str, Any], category: str = 'generic') -> Dict[str, Any]:
    """
    Filter invalid nodes and remove edges pointing to invalid nodes.
    
    Args:
        graph: dict with 'nodes' and 'edges'
        category: node category for selecting filter rules
    
    Returns:
        Filtered graph
    """
    if not graph or 'nodes' not in graph:
        return graph
    
    # 1. Filter nodes
    valid_nodes = []
    valid_node_ids = set()
    
    for node in graph.get('nodes', []):
        if is_valid_node(node, category):
            valid_nodes.append(node)
            valid_node_ids.add(str(node.get('id', '')))
    
    # 2. Filter edges: keep only edges connecting valid nodes
    valid_edges = []
    for edge in graph.get('edges', []):
        src = str(edge.get('from', ''))
        dst = str(edge.get('to', ''))
        
        # Keep edge only if both endpoints are valid
        if src in valid_node_ids and dst in valid_node_ids:
            valid_edges.append(edge)
    
    return {
        'nodes': valid_nodes,
        'edges': valid_edges
    }


# =============================================================================
# Evolution-Aware View: API evolution sub-graph
# =============================================================================

def _extract_imports(code: str) -> List[str]:
    """
    Extract use-statement dependency list from Reference Code.
    
    Example:
        use std::os::unix::fs::chown;  -> ["std::os::unix::fs::chown"]
        use std::path::Path;           -> ["std::path::Path"]
    """
    if not code:
        return []
    
    imports = []
    # Match use xxx::yyy::zzz;
    use_statements = re.findall(r'use\s+([\w:]+(?:::\{[^}]+\})?)\s*;', code)
    
    for stmt in use_statements:
        # Handle use std::{A, B} form
        if '{' in stmt:
            base = stmt.split('::')[:-1]
            items = re.findall(r'\w+', stmt.split('{')[1].split('}')[0])
            for item in items:
                imports.append('::'.join(base + [item]))
        else:
            imports.append(stmt)
    
    return imports[:5]  # Keep at most 5 key dependencies


def _extract_source_code_structure(source_code: str, api_name: str, role: str) -> Dict[str, Any]:
    """
    Extract structural information from source_code (functions, types, calls).
    
    Core output:
    - Function/method definition nodes
    - Parameter and type nodes
    - Call-chain nodes
    - Structural relation edges
    
    Args:
        source_code: source code string
        api_name: API name
        role: 'old', 'new', or 'current'
    
    Returns:
        Dict with nodes and edges
    """
    import re
    
    nodes = []
    edges = []
    first_func_id = None  # Track first function for method_call edges
    
    if not source_code or not source_code.strip():
        return {'nodes': nodes, 'edges': edges}
    
    # 1. Extract function/method definitions
    func_pattern = re.compile(
        r'^(?:pub\s+)?(?:unsafe\s+)?(?:const\s+)?(?:async\s+)?'
        r'fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^\{;]+))?',
        re.MULTILINE
    )
    
    for match in func_pattern.finditer(source_code):
        func_name = match.group(1)
        params_str = match.group(2).strip()
        return_type = match.group(3).strip() if match.group(3) else 'void'
        
        # Add function node
        func_node_id = f'func_{func_name}'
        if first_func_id is None:
            first_func_id = func_node_id
        nodes.append({
            'id': func_node_id,
            'type': 'function_def',
            'name': func_name,
            'return_type': return_type.strip(),
            'role': role,
            'source': 'source_code'  # Provenance marker
        })
        
        # Parse parameters
        if params_str:
            params = _parse_params_simple(params_str)
            for i, (param_name, param_type) in enumerate(params):
                param_node_id = f'param_{func_name}_{param_name}'
                nodes.append({
                    'id': param_node_id,
                    'type': 'param',
                    'name': param_name,
                    'param_type': param_type,
                    'position': i,
                    'role': role,
                    'function': func_name,
                })
                edges.append({
                    'from': param_node_id,
                    'to': func_node_id,
                    'type': 'param_of'
                })
    
    # 2. Extract type definitions (struct, enum, trait)
    type_patterns = [
        (r'(?:pub\s+)?struct\s+(\w+)(?:<[^>]*>)?', 'struct_def'),
        (r'(?:pub\s+)?enum\s+(\w+)(?:<[^>]*>)?', 'enum_def'),
        (r'(?:pub\s+)?trait\s+(\w+)(?:<[^>]*>)?', 'trait_def'),
        (r'(?:pub\s+)?type\s+(\w+)(?:<[^>]*>)?\s*=', 'type_alias'),
    ]
    
    for pattern, node_type in type_patterns:
        for match in re.finditer(pattern, source_code):
            type_name = match.group(1)
            type_node_id = f'{node_type}_{type_name}'
            nodes.append({
                'id': type_node_id,
                'type': node_type,
                'name': type_name,
                'role': role
            })
    
    # 3. Extract impl blocks
    impl_id_counts = {}
    impl_pattern = re.compile(
        r'impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)(?:<[^>]*>)?',
        re.MULTILINE
    )
    
    first_impl_id = None  # Track first impl node for import edges
    for match in impl_pattern.finditer(source_code):
        trait_name = match.group(1)  # May be None
        type_name = match.group(2)
        
        impl_node_id = f'impl_{type_name}'
        if trait_name:
            impl_node_id = f'impl_{trait_name}_for_{type_name}'
        count = impl_id_counts.get(impl_node_id, 0)
        impl_id_counts[impl_node_id] = count + 1
        if count:
            impl_node_id = f'{impl_node_id}_{count}'
        
        if first_impl_id is None:
            first_impl_id = impl_node_id
        
        nodes.append({
            'id': impl_node_id,
            'type': 'impl_block',
            'target_type': type_name,
            'trait_impl': trait_name,
            'role': role
        })
    
    # 4. Extract use statements (dependencies)
    use_pattern = re.compile(r'use\s+([^;]+);')
    imports = []
    for match in use_pattern.finditer(source_code):
        use_path = match.group(1).strip()
        imports.append(use_path)
    
    if imports:
        # Create aggregated imports node
        nodes.append({
            'id': 'imports',
            'type': 'import_group',
            'imports': imports[:10],  # Keep at most 10
            'role': role
        })
        # Connect to impl or first function to avoid orphan
        if first_impl_id:
            edges.append({'from': 'imports', 'to': first_impl_id, 'type': 'imports_for'})
        elif nodes:  # Find first function/struct
            for node in nodes:
                if node.get('type') in ['function_def', 'struct_def']:
                    edges.append({'from': 'imports', 'to': node['id'], 'type': 'imports_for'})
                    break
    
    # 5. Extract method calls (API-related only)
    call_pattern = re.compile(r'\.(\w+)\s*\(')
    method_calls = _dedupe_preserve_order(call_pattern.findall(source_code))[:8]  # Limit count
    
    # Filter: keep only potentially evolution-related calls
    # Exclude generic methods (new, clone, join, etc.)
    generic_methods = {'new', 'clone', 'to_string', 'into', 'from', 'map', 
                      'unwrap', 'expect', 'is_some', 'is_none', 'iter', 'collect',
                      'push', 'pop', 'get', 'set', 'len', 'is_empty', 'join'}
    
    if method_calls:
        for call in method_calls:
            # Skip generic methods (unless it is the API name itself)
            if call in generic_methods and call != api_name:
                continue
            
            call_node_id = f'call_{call}'
            nodes.append({
                'id': call_node_id,
                'type': 'method_call',
                'name': call,
                'role': role,
                'source': 'inferred'  # Provenance marker
            })
            # Connect to first function to avoid orphan
            if first_func_id:
                edges.append({
                    'from': call_node_id,
                    'to': first_func_id,
                    'type': 'invoked_in'
                })
    
    # 6. Extract macro invocations
    macro_pattern = re.compile(r'(\w+)!')
    macros = _dedupe_preserve_order(macro_pattern.findall(source_code))[:5]
    
    for macro in macros:
        if macro not in ['cfg', 'doc', 'test', 'derive', 'allow', 'warn', 'deny']:  # Skip common attribute macros
            nodes.append({
                'id': f'macro_{macro}',
                'type': 'macro_call',
                'name': macro,
                'role': role
            })
    
    # Filter source structure to reduce noise
    result = {'nodes': nodes, 'edges': edges}
    return filter_graph_nodes(result, category='generic')


def _parse_params_simple(params_str: str) -> List[Tuple[str, str]]:
    """
    Simple parameter parsing: extract name and type.
    """
    params = []
    if not params_str:
        return params
    
    seen_self = False
    for raw in _split_params(params_str):
        raw = raw.strip()
        if not raw:
            continue
        name, ptype = _extract_param_name_type(raw)
        if not name:
            continue
        if name == 'self':
            if not seen_self:
                params.append((name, ptype))
                seen_self = True
            continue
        params.append((name, ptype))
    
    return params


def _extract_param_name_type(param: str) -> Tuple[str, str]:
    """
    Extract parameter name and type from a raw string.
    """
    # Handle self parameter
    if param.strip() in ['self', '&self', '&mut self', 'mut self']:
        return ('self', 'Self')
    
    # Match name: type pattern
    match = re.match(r'^(\w+)\s*:\s*(.+)$', param.strip())
    if match:
        return (match.group(1), match.group(2).strip())
    
    return ('', '')


def _add_evolution_edges(old_structure: Dict, new_structure: Dict, all_edges: List):
    """
    Add evolution edges connecting old and new source structures.
    
    Strategy:
    1. evolves_to edges between same-name functions
    2. type_evolves edges between same-name types
    3. param_changes edges for parameter diffs
    """
    old_nodes = {n['id']: n for n in old_structure.get('nodes', [])}
    new_nodes = {n['id']: n for n in new_structure.get('nodes', [])}
    
    # 1. Connect same-name functions
    for old_id, old_node in old_nodes.items():
        if old_node.get('type') == 'function_def':
            func_name = old_node.get('name', '')
            # Find same-name function in new code
            for new_id, new_node in new_nodes.items():
                if new_node.get('type') == 'function_def' and new_node.get('name') == func_name:
                    all_edges.append({
                        'from': f'old_{old_id}',
                        'to': f'new_{new_id}',
                        'type': 'evolves_to'
                    })
                    
                    # Check return type change
                    old_return = old_node.get('return_type', '')
                    new_return = new_node.get('return_type', '')
                    if old_return != new_return:
                        all_edges.append({
                            'from': f'old_{old_id}',
                            'to': f'new_{new_id}',
                            'type': 'return_type_changed',
                            'old_type': old_return,
                            'new_type': new_return
                        })
    
    # 2. Connect same-name type definitions
    type_kinds = ['struct_def', 'enum_def', 'trait_def', 'type_alias']
    for old_id, old_node in old_nodes.items():
        if old_node.get('type') in type_kinds:
            type_name = old_node.get('name', '')
            for new_id, new_node in new_nodes.items():
                if new_node.get('type') in type_kinds and new_node.get('name') == type_name:
                    all_edges.append({
                        'from': f'old_{old_id}',
                        'to': f'new_{new_id}',
                        'type': 'type_evolves'
                    })
    
    def _param_func_from_id(param_id: str) -> str:
        if param_id.startswith('param_'):
            tail = param_id[len('param_'):]
            parts = tail.rsplit('_', 1)
            if len(parts) == 2:
                return parts[0]
        return ''

    # 3. Connect parameter changes within same function
    for old_id, old_node in old_nodes.items():
        if old_node.get('type') == 'param':
            param_name = old_node.get('name', '')
            param_func = old_node.get('function') or _param_func_from_id(old_id)
            
            for new_id, new_node in new_nodes.items():
                if new_node.get('type') == 'param':
                    new_param_name = new_node.get('name', '')
                    new_func = new_node.get('function') or _param_func_from_id(new_id)
                    
                    # Same function, same parameter name
                    if param_func == new_func and param_name == new_param_name:
                        old_type = old_node.get('param_type', '')
                        new_type = new_node.get('param_type', '')
                        
                        if old_type != new_type:
                            all_edges.append({
                                'from': f'old_{old_id}',
                                'to': f'new_{new_id}',
                                'type': 'param_type_changed',
                                'old_type': old_type,
                                'new_type': new_type
                            })
                        else:
                            all_edges.append({
                                'from': f'old_{old_id}',
                                'to': f'new_{new_id}',
                                'type': 'param_preserved'
                            })


def extract_api_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build Evolution-Aware sub-graph from source_code and old_source_code.

    Core strategy:
    1. Differentiated strategy per change_type
    2. Focus on core API evolution differences
    3. Extract import dependencies from Reference Code
    
    Dispatch by change_type:
    - stabilized: availability change + imports
    - signature: signature diff + imports
    - deprecated: replacement API + imports
    - implicit: call sequence diff + imports
    """
    change_type = sample.get('change_type', '')
    
    # Dispatch by change_type
    if change_type == 'stabilized':
        return _extract_stabilized_evolution(sample)
    elif change_type == 'signature':
        return _extract_signature_evolution(sample)
    elif change_type == 'deprecated':
        return _extract_deprecated_evolution(sample)
    elif change_type == 'implicit':
        return _extract_implicit_evolution(sample)
    else:
        # Fallback: try signature analysis
        return _extract_signature_evolution(sample)


def _extract_signature_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    signature: Signature evolution (Evolution-Aware View).
    
    Data: 246 samples, 233 (95%) have old_source_code, 13 signature-only.
    Strategy: requires old_source_code for genuine evolution graph.
    Note: signature and source_code are both new-version info.
    """
    old_code = sample.get('old_source_code', '')
    new_code = sample.get('source_code', '')
    api_name = sample.get('name', '')
    code = sample.get('code', '')
    signature = sample.get('signature', '')
    
    # Require old_source_code for genuine old vs new comparison
    # signature and source_code are both new-version and cannot approximate old
    if not old_code:
        return None
    
    # Extract imports
    imports = _extract_imports(code)

    # Non-function signatures (impl/struct/enum/const/type)
    old_api_parsed = parse_api_signature(old_code)
    new_api_parsed = parse_api_signature(new_code if new_code else signature)
    if old_api_parsed.get('kind') != 'fn' or new_api_parsed.get('kind') != 'fn':
        changes = []
        def add_change(kind: str, old_val: Any, new_val: Any, value: Optional[str] = None):
            if old_val != new_val:
                item = {"kind": kind, "old": old_val, "new": new_val}
                if value:
                    item["value"] = value
                changes.append(item)

        add_change("kind", old_api_parsed.get("kind"), new_api_parsed.get("kind"))
        add_change("name", old_api_parsed.get("name"), new_api_parsed.get("name"))
        add_change("visibility", old_api_parsed.get("visibility"), new_api_parsed.get("visibility"))
        add_change("generics", old_api_parsed.get("generics"), new_api_parsed.get("generics"),
                   value=f"{old_api_parsed.get('generics')} -> {new_api_parsed.get('generics')}")
        add_change("impl_trait", old_api_parsed.get("impl_trait"), new_api_parsed.get("impl_trait"))
        add_change("impl_for", old_api_parsed.get("impl_for"), new_api_parsed.get("impl_for"))
        add_change("return_type", old_api_parsed.get("return_type"), new_api_parsed.get("return_type"),
                   value=f"{old_api_parsed.get('return_type')} -> {new_api_parsed.get('return_type')}")
        add_change("const", old_api_parsed.get("is_const"), new_api_parsed.get("is_const"),
                   value=f"{old_api_parsed.get('is_const')} -> {new_api_parsed.get('is_const')}")
        add_change("unsafe", old_api_parsed.get("is_unsafe"), new_api_parsed.get("is_unsafe"),
                   value=f"{old_api_parsed.get('is_unsafe')} -> {new_api_parsed.get('is_unsafe')}")

        # old_code existence already verified above

        nodes = [
            {
                "id": "old",
                "type": old_api_parsed.get("kind", "signature"),
                "api": api_name,
                "name": old_api_parsed.get("name"),
                "visibility": old_api_parsed.get("visibility"),
                "generics": old_api_parsed.get("generics"),
                "impl_trait": old_api_parsed.get("impl_trait"),
                "impl_for": old_api_parsed.get("impl_for"),
                "return_type": old_api_parsed.get("return_type"),
                "is_const": old_api_parsed.get("is_const"),
                "is_unsafe": old_api_parsed.get("is_unsafe"),
                "role": "old",
            },
            {
                "id": "new",
                "type": new_api_parsed.get("kind", "signature"),
                "api": api_name,
                "name": new_api_parsed.get("name"),
                "visibility": new_api_parsed.get("visibility"),
                "generics": new_api_parsed.get("generics"),
                "impl_trait": new_api_parsed.get("impl_trait"),
                "impl_for": new_api_parsed.get("impl_for"),
                "return_type": new_api_parsed.get("return_type"),
                "is_const": new_api_parsed.get("is_const"),
                "is_unsafe": new_api_parsed.get("is_unsafe"),
                "role": "new",
            },
        ]
        edges = []

        for i, change in enumerate(changes):
            cid = f"c{i}"
            nodes.append({
                "id": cid,
                "type": "change",
                **change
            })
            edges.append({"from": "old", "to": cid, "type": "has_change"})
            edges.append({"from": cid, "to": "new", "type": "modifies"})

        edges.append({
            "from": "old",
            "to": "new",
            "type": "evolves_to"
        })

        if imports:
            nodes.append({"id": "code_imports", "type": "dependencies", "items": imports})
            edges.append({"from": "new", "to": "code_imports", "type": "requires"})

        result = {"nodes": nodes, "edges": edges}
        return filter_graph_nodes(result, category='api_evolution')
    
    # Enhanced full signature extraction
    def extract_full_signature(code_str):
        """
        Extract full signature info:
        - visibility: pub, pub(crate), private
        - modifiers: const, unsafe, async
        - name: function name
        - generics: generic parameters
        - params: function parameters
        - return_type: return type
        - where_clause: where clause
        """
        result = {
            'visibility': None,
            'is_const': False,
            'is_unsafe': False,
            'is_async': False,
            'name': None,
            'generics': [],
            'params': [],
            'return_type': None,
            'where_clause': None,
            'raw_signature': code_str.strip()
        }
        
        # 0. Preprocess: extract fn-containing line from multiline code
        if '\n' in code_str:
            lines = code_str.split('\n')
            sig_line = ''
            for line in lines:
                # Find fn-containing line (skip attribute lines)
                if 'fn ' in line and '(' in line and not line.strip().startswith('#'):
                    sig_line = line
                    # Check for where clause in subsequent lines
                    idx = lines.index(line)
                    if idx + 1 < len(lines):
                        for next_line in lines[idx+1:]:
                            if 'where' in next_line:
                                sig_line += ' ' + next_line
                                break
                            if '{' in next_line:
                                break
                    break
            if sig_line:
                code_str = sig_line
            else:
                # No fn line found; may be struct/const etc.
                for line in lines:
                    if not line.strip().startswith('#'):
                        code_str = line
                        break
        
        # 1. Extract visibility
        if code_str.strip().startswith('pub(crate)'):
            result['visibility'] = 'pub(crate)'
            code_str = code_str.replace('pub(crate)', '', 1).strip()
        elif code_str.strip().startswith('pub'):
            result['visibility'] = 'pub'
            code_str = code_str.replace('pub', '', 1).strip()
        else:
            result['visibility'] = 'private'
        
        # 2. Extract modifiers
        if 'const fn' in code_str:
            result['is_const'] = True
        if 'unsafe fn' in code_str or code_str.strip().startswith('unsafe'):
            result['is_unsafe'] = True
        if 'async fn' in code_str:
            result['is_async'] = True
        
        # 3. Extract where clause
        where_match = re.search(r'\bwhere\b(.+?)(?:\{|$)', code_str)
        if where_match:
            result['where_clause'] = where_match.group(1).strip()
            # Remove where clause for subsequent parsing
            code_str = code_str[:where_match.start()]
        
        # 4. Extract function name and generics
        fn_match = re.search(r'\b(?:const\s+)?(?:unsafe\s+)?(?:async\s+)?fn\s+(\w+)\s*(<([^>]+)>)?\s*\(', code_str)
        if fn_match:
            result['name'] = fn_match.group(1)
            if fn_match.group(3):  # Has generics
                generic_str = fn_match.group(3)
                for g in re.split(r',\s*(?![^<>]*>)', generic_str):
                    g = g.strip()
                    if ':' in g and '::' not in g:
                        parts = g.split(':', 1)
                        result['generics'].append({'name': parts[0].strip(), 'bound': parts[1].strip()})
                    else:
                        result['generics'].append({'name': g, 'bound': None})
        
        # 5. Extract parameters
        params_str = _extract_params(code_str)
        if params_str is not None:
            params_str = params_str.strip()
            if params_str:
                for raw in _split_params(params_str):
                    param = raw.strip()
                    if not param:
                        continue
                    if param in ('self', '&self', '&mut self', 'mut self'):
                        if param == '&mut self':
                            result['params'].append(('self', '&mut Self'))
                        elif param == '&self':
                            result['params'].append(('self', '&Self'))
                        elif param == 'mut self':
                            result['params'].append(('self', 'mut Self'))
                        else:
                            result['params'].append(('self', 'Self'))
                        continue
                    if ':' in param:
                        name, typ = param.split(':', 1)
                        result['params'].append((name.strip(), typ.strip()))
        
        # 6. Extract return type
        ret_match = re.search(r'->\s*(.+?)(?:\s*(?:where|\{)|$)', code_str)
        if ret_match:
            ret_type = ret_match.group(1).strip()
            # Strip braces and trailing content
            brace_idx = ret_type.find('{')
            if brace_idx != -1:
                ret_type = ret_type[:brace_idx].strip()
            result['return_type'] = ret_type
        
        return result
    
    # Parse old and new signatures
    old_sig_info = extract_full_signature(old_code)
    new_sig_info = extract_full_signature(new_code if new_code else signature)
    
    # Legacy simplified extraction for compatibility
    old_params = old_sig_info['params']
    old_generics = old_sig_info['generics']
    new_params = new_sig_info['params']
    new_generics = new_sig_info['generics']
    
    # Detect const modifier change (preserve original logic)
    old_is_const = 'const fn' in old_code or (old_code.startswith('const') and '(' not in old_code.split('=')[0])
    new_is_const = 'const fn' in (new_code if new_code else signature) or ((new_code if new_code else signature).startswith('const') and '(' not in (new_code if new_code else signature).split('=')[0])
    const_changed = old_is_const != new_is_const
    
    # If no params, try extracting return type diff from signature
    if not old_params and not new_params:
        # Extract return type
        old_ret = re.search(r'->\s*([^{]+)', old_code)
        new_ret = re.search(r'->\s*([^{]+)', new_code if new_code else signature)
        if old_ret and new_ret:
            old_ret_type = old_ret.group(1).strip()
            new_ret_type = new_ret.group(1).strip()
            if old_ret_type != new_ret_type:
                # At least a return type change exists
                old_params = [('return', old_ret_type)]
                new_params = [('return', new_ret_type)]
        
        # Keep if generics changed, even without param changes
        if not old_params and not new_params and old_generics == new_generics:
            # Final fallback: unstructured text diff
            # Provide optimized text diff parsing
            new_text = (new_code if new_code else signature).strip()
            old_text = old_code.strip()
            
            # Extract signature line (first fn/impl/struct/const line)
            def extract_sig_line(text):
                for line in text.split('\n'):
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        if any(kw in stripped for kw in ['fn ', 'impl ', 'struct ', 'const ', 'enum ', 'type ']):
                            return stripped
                return text.split('\n')[0].strip() if text else ''
            
            old_sig_line = extract_sig_line(old_text)
            new_sig_line = extract_sig_line(new_text)
            
            # If signature lines differ, attempt structured parse
            if old_sig_line and new_sig_line and old_sig_line != new_sig_line:
                # Identify change type
                change_kind = "signature_text_diff"
                changes = []
                
                # 1. const modifier change
                old_has_const = 'const fn' in old_sig_line
                new_has_const = 'const fn' in new_sig_line
                if old_has_const != new_has_const:
                    change_kind = "const_modifier_changed"
                    changes.append({
                        "modifier": "const",
                        "old": old_has_const,
                        "new": new_has_const
                    })
                
                # 2. impl block change
                old_is_impl = old_sig_line.strip().startswith('impl')
                new_is_impl = new_sig_line.strip().startswith('impl')
                if old_is_impl or new_is_impl:
                    if old_is_impl and new_is_impl:
                        # impl const Default -> impl Default
                        if 'const' in old_sig_line and 'const' not in new_sig_line:
                            change_kind = "impl_const_removed"
                            changes.append({"modifier": "const", "old": True, "new": False})
                        elif 'const' not in old_sig_line and 'const' in new_sig_line:
                            change_kind = "impl_const_added"
                            changes.append({"modifier": "const", "old": False, "new": True})
                        # Type change: impl Default for TypeA -> TypeB
                        elif 'for' in old_sig_line and 'for' in new_sig_line:
                            old_type = old_sig_line.split('for')[-1].strip().rstrip('{').strip()
                            new_type = new_sig_line.split('for')[-1].strip().rstrip('{').strip()
                            if old_type != new_type:
                                change_kind = "impl_type_changed"
                                changes.append({
                                    "target_type": {"old": old_type, "new": new_type}
                                })
                    elif old_is_impl and not new_is_impl:
                        # impl -> fn conversion
                        change_kind = "impl_to_function"
                        changes.append({"from": "impl", "to": "fn"})
                    elif not old_is_impl and new_is_impl:
                        # fn -> impl conversion
                        change_kind = "function_to_impl"
                        changes.append({"from": "fn", "to": "impl"})
                
                nodes = [
                    {"id": "old", "type": "text_sig", "api": api_name, "signature": old_sig_line, "role": "old"},
                    {"id": "new", "type": "text_sig", "api": api_name, "signature": new_sig_line, "role": "new"},
                    {"id": "c0", "type": "change", "kind": change_kind, 
                     "old": old_sig_line, "new": new_sig_line, "changes": changes}
                ]
                edges = [
                    {"from": "old", "to": "new", "type": "evolves"},
                    {"from": "c0", "to": "new", "type": "modifies"}
                ]
                result = {"nodes": nodes, "edges": edges}
                return filter_graph_nodes(result, category='api_evolution')
            
            return None
    
    # Build full signature evolution graph: record all dimension changes
    nodes = [
        {
            "id": "old", 
            "type": "sig", 
            "api": api_name, 
            "params": [{"name": p[0], "type": p[1].strip()} for p in old_params[:3]], 
            "generics": old_generics,
            "visibility": old_sig_info['visibility'],
            "is_const": old_sig_info['is_const'],
            "is_unsafe": old_sig_info['is_unsafe'],
            "is_async": old_sig_info['is_async'],
            "return_type": old_sig_info['return_type'],
            "role": "old"
        },
        {
            "id": "new", 
            "type": "sig", 
            "api": api_name, 
            "params": [{"name": p[0], "type": p[1].strip()} for p in new_params[:3]], 
            "generics": new_generics,
            "visibility": new_sig_info['visibility'],
            "is_const": new_sig_info['is_const'],
            "is_unsafe": new_sig_info['is_unsafe'],
            "is_async": new_sig_info['is_async'],
            "return_type": new_sig_info['return_type'],
            "role": "new"
        },
    ]
    edges = [{"from": "old", "to": "new", "type": "evolves"}]
    
    # Add change nodes: detect all dimension changes
    changes = []
    old_param_names = {p[0]: p[1].strip() for p in old_params}
    new_param_names = {p[0]: p[1].strip() for p in new_params}
    
    # 1. Detect visibility change
    if old_sig_info['visibility'] != new_sig_info['visibility']:
        changes.append({
            "kind": "visibility_changed",
            "old": old_sig_info['visibility'],
            "new": new_sig_info['visibility']
        })
    
    # 2. Detect unsafe modifier change
    if old_sig_info['is_unsafe'] != new_sig_info['is_unsafe']:
        changes.append({
            "kind": "unsafe_modifier_changed",
            "old": old_sig_info['is_unsafe'],
            "new": new_sig_info['is_unsafe']
        })
    
    # 3. Detect async modifier change
    if old_sig_info['is_async'] != new_sig_info['is_async']:
        changes.append({
            "kind": "async_modifier_changed",
            "old": old_sig_info['is_async'],
            "new": new_sig_info['is_async']
        })
    
    # 4. Detect return type change
    if old_sig_info['return_type'] != new_sig_info['return_type']:
        changes.append({
            "kind": "return_type_changed",
            "old": old_sig_info['return_type'],
            "new": new_sig_info['return_type']
        })
    
    # 5. Detect where clause change
    if old_sig_info['where_clause'] != new_sig_info['where_clause']:
        changes.append({
            "kind": "where_clause_changed",
            "old": old_sig_info['where_clause'],
            "new": new_sig_info['where_clause']
        })
    
    # 6. Detect parameter add/remove
    old_param_set = set(old_param_names.keys())
    new_param_set = set(new_param_names.keys())
    added_params = sorted(new_param_set - old_param_set)
    removed_params = sorted(old_param_set - new_param_set)
    
    if added_params:
        changes.append({
            "kind": "param_added",
            "params": [{"name": p, "type": new_param_names[p]} for p in added_params]
        })
    if removed_params:
        changes.append({
            "kind": "param_removed",
            "params": [{"name": p, "type": old_param_names[p]} for p in removed_params]
        })
    
    # 7. Detect generics change
    if old_generics != new_generics:
        old_gen_names = {g['name']: g.get('bound') for g in old_generics}
        new_gen_names = {g['name']: g.get('bound') for g in new_generics}
        
        # Added generics
        added_gens = [g for g in new_generics if g['name'] not in old_gen_names]
        added_gens.sort(key=lambda g: g.get('name', ''))
        # Removed generics
        removed_gens = [g for g in old_generics if g['name'] not in new_gen_names]
        removed_gens.sort(key=lambda g: g.get('name', ''))
        # Bound changes
        changed_gens = []
        for name in sorted(set(old_gen_names.keys()) & set(new_gen_names.keys())):
            if old_gen_names[name] != new_gen_names[name]:
                changed_gens.append({'name': name, 'old': old_gen_names[name], 'new': new_gen_names[name]})
        changed_gens.sort(key=lambda g: g.get('name', ''))
        
        if added_gens:
            changes.append({"kind": "generic_added", "generics": added_gens})
        if removed_gens:
            changes.append({"kind": "generic_removed", "generics": removed_gens})
        if changed_gens:
            changes.append({"kind": "generic_bound_changed", "changes": changed_gens})
    
    # 8. Detect const modifier change
    if const_changed:
        changes.append({"kind": "const_modifier", "old": old_is_const, "new": new_is_const})
    
    # 9. Detect rename (name changed, type preserved)
    if len(old_param_names) == len(new_param_names) and set(old_param_names.values()) == set(new_param_names.values()):
        old_names = list(old_param_names.keys())
        new_names = list(new_param_names.keys())
        for i, (o, n) in enumerate(zip(old_names, new_names)):
            if o != n and old_param_names.get(o) == new_param_names.get(n):
                changes.append({"kind": "rename", "old": o, "new": n})
    
    # 10. Detect parameter type change
    for param in sorted(set(old_param_names.keys()) & set(new_param_names.keys())):
        if old_param_names[param] != new_param_names[param]:
            changes.append({"kind": "type", "param": param, "old": old_param_names[param], "new": new_param_names[param]})
    
    # =========================================================================
    # Evolution-Aware View: signature evolution sub-graph
    # Core: actual parameter name/type change patterns
    # =========================================================================
    
    # old_code existence already verified above
    
    # Add change nodes
    for i, change in enumerate(changes):
        cid = f"c{i}"
        change_kind = change.get('kind', '')
        nodes.append({
            "id": cid, 
            "type": "change", 
            **change
        })
        edges.append({"from": "old", "to": cid, "type": "has_change"})
        edges.append({"from": cid, "to": "new", "type": "modifies"})
    
    # Add evolution edge
    edges.append({
        "from": "old", 
        "to": "new", 
        "type": "evolves_to"
    })
    
    # If no changes detected
    if not changes:
        nodes.append({
            "id": "c0",
            "type": "change",
            "kind": "signature_unchanged",
            "note": "No detectable differences"
        })
        edges.append({"from": "old", "to": "c0", "type": "annotates"})
    
    # Add imports node
    if imports:
        nodes.append({"id": "code_imports", "type": "dependencies", "items": imports})
        edges.append({"from": "new", "to": "code_imports", "type": "requires"})
    
    result = {"nodes": nodes, "edges": edges}
    return filter_graph_nodes(result, category='api_evolution')


def _extract_implicit_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    implicit: Implementation behaviour change (Evolution-Aware View).
    
    Data: 217 samples, all have old_source_code, mainly internal call changes.
    Strategy: retain behaviour/unsafe evidence only; avoid implementation noise.
    """
    old_code = sample.get('old_source_code', '')
    new_code = sample.get('source_code', '')
    api_name = sample.get('name', '')
    code = sample.get('code', '')
    
    if not old_code or not new_code:
        return None
    
    # Extract imports
    imports = _extract_imports(code)
    
    # Change features (lightweight: only whether change occurred)
    old_methods = set(re.findall(r'\.([a-z_][a-z0-9_]*)\s*\(', old_code))
    new_methods = set(re.findall(r'\.([a-z_][a-z0-9_]*)\s*\(', new_code))
    old_intrinsics = set(re.findall(r'intrinsics::([a-z0-9_]+)', old_code))
    new_intrinsics = set(re.findall(r'intrinsics::([a-z0-9_]+)', new_code))
    unsafe_changed = ('unsafe' in old_code) != ('unsafe' in new_code)
    
    changes = []
    if old_methods != new_methods or old_intrinsics != new_intrinsics:
        changes.append({
            "kind": "behavior_changed",
        })
    if unsafe_changed:
        changes.append({
            "kind": "unsafe_changed",
            "old": 'unsafe' in old_code,
            "new": 'unsafe' in new_code,
        })
    changes = changes[:2]
    
    # =========================================================================
    # Evolution-Aware View: behaviour evolution sub-graph
    # Data: 217 samples, 100% have old_source_code comparison
    # Core: fine-grained implementation changes
    # =========================================================================
    
    nodes = [
        {"id": "old", "type": "impl", "api": api_name, "role": "old"},
        {"id": "new", "type": "impl", "api": api_name, "role": "new"},
    ]
    
    edges = [{
        "from": "old", 
        "to": "new", 
        "type": "refactors"
    }]
    
    for i, change in enumerate(changes):
        cid = f"c{i}"
        nodes.append({"id": cid, "type": "change", **change})
        edges.append({"from": "old", "to": cid, "type": "has_change"})
        edges.append({"from": cid, "to": "new", "type": "modifies"})
    
    # Add imports node
    if imports:
        nodes.append({"id": "code_imports", "type": "dependencies", "items": imports})
        edges.append({"from": "new", "to": "code_imports", "type": "requires"})
    
    result = {"nodes": nodes, "edges": edges}
    return filter_graph_nodes(result, category='api_evolution')


def _extract_replacement_from_code(api_name: str, code: str, sample: Dict[str, Any]) -> Optional[str]:
    """
    Extract replacement API from Reference Code.
    
    Strategy:
    1. Type alias in old_source_code (most reliable)
    2. Explicit replacement name from deprecated note
    3. Primary type from use statements in code
    4. Method calls from code
    """
    # 1. Type alias (most reliable evidence)
    old_code = sample.get('old_source_code', '')
    if 'pub type' in old_code:
        alias_match = re.search(r'pub type \w+[^=]*=\s*([^;{]+)', old_code)
        if alias_match:
            return alias_match.group(1).strip()
    
    # 2. Extract from deprecated note
    source_code = sample.get('source_code', '')
    documentation = sample.get('documentation', '')
    dep_note_match = re.search(r'#\[deprecated[^\]]*note\s*=\s*"([^"]+)"', source_code)
    if not dep_note_match and documentation:
        dep_note_match = re.search(r'note\s*=\s*"([^"]+)"', documentation)
    replacement = None
    if dep_note_match:
        note = dep_note_match.group(1)
        
        # Extract explicit replacement API name
        # Patterns: "use XXX" / "XXX instead" / "superseded by `XXX`"
        patterns = [
            r'superseded by `([\w:]+)`',
            r'[Uu]se `?([\w:]+)(?:\(\))?`?',
            r'use (?:the )?([\w:]+) (?:impl|instead)',
        ]
        for pat in patterns:
            match = re.search(pat, note)
            if match:
                candidate = match.group(1)
                if candidate in ['this', 'that', 'it', 'the', 'a', 'an', 'or']:
                    continue
                replacement = candidate
                break

    if not replacement and documentation:
        # Extract replacement API from documentation body
        doc_patterns = [
            r'[Uu]se `?([\w:]+)(?:\(\))?`? instead',
            r'replaced by `?([\w:]+)`?',
            r'superseded by `?([\w:]+)`?',
        ]
        for pat in doc_patterns:
            match = re.search(pat, documentation)
            if match:
                candidate = match.group(1)
                if candidate in ['this', 'that', 'it', 'the', 'a', 'an', 'or']:
                    continue
                replacement = candidate
                break

    if replacement:
        return replacement
    
    # 3. Extract primary type from use statements
    # Match two forms: use path::Type and use path::{Type1, Type2}
    use_imports = []
    # Form 1: use path::Type;
    use_imports.extend(re.findall(r'use [^;{]+::([\w]+)\s*;', code))
    # Form 2: use path::{Type1, Type2};
    for match in re.finditer(r'use [^;]+\{([^}]+)\}', code):
        types = match.group(1).split(',')
        use_imports.extend([t.strip() for t in types if t.strip()])
    
    # Filter: capitalised, not generic types
    candidates = [u for u in use_imports if u and u[0].isupper() 
                  and u not in ['Self', 'String', 'Vec', 'Option', 'Result', 'Box', 'Arc', 'Rc', 'Error']]
    
    # Prefer: struct/enum/trait definitions or specific use-imported types
    # Find struct/enum defined in code (custom replacement)
    defined_types = re.findall(r'(?:struct|enum|trait)\s+(\w+)', code)
    if defined_types:
        # Prefer use-imported over self-defined types (migration target)
        if candidates:
            # Exclude self-defined type names
            imported_types = [c for c in candidates if c not in defined_types]
            if imported_types:
                candidates = imported_types
    
    # If exactly one candidate type, use it
    unique_candidates = _dedupe_preserve_order(candidates)
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    
    # 4. For method-style APIs, extract method calls from code
    if api_name and api_name[0].islower():
        # Find most frequent method calls
        method_calls = re.findall(r'\.(\w+)\(', code)
        if method_calls:
            from collections import Counter
            method_counts = Counter(method_calls)
            # Exclude generic methods
            common_methods = {'new', 'clone', 'to_string', 'map', 'unwrap', 'expect', 'is_some', 'is_none'}
            filtered_methods = [(m, c) for m, c in method_counts.items() if m not in common_methods]
            if filtered_methods:
                filtered_methods.sort(key=lambda x: (-x[1], x[0]))
                return filtered_methods[0][0]
    
    # 5. Multiple candidates: pick most likely
    if candidates:
        # Prefer types appearing multiple times in code
        from collections import Counter
        type_counts = Counter(candidates)
        return sorted(type_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    
    # Cannot extract
    return None


def _extract_declared_name(code: str) -> Optional[str]:
    """
    Extract declared API name from source text (for deprecated old/new).
    """
    if not code:
        return None

    patterns = [
        r'\b(?:pub\s+)?type\s+(\w+)\b',
        r'\b(?:pub\s+)?struct\s+(\w+)\b',
        r'\b(?:pub\s+)?enum\s+(\w+)\b',
        r'\b(?:pub\s+)?const\s+([A-Z_][A-Z0-9_]*)\b',
        r'\b(?:pub\s+)?fn\s+(\w+)\b',
    ]
    for pat in patterns:
        match = re.search(pat, code)
        if match:
            return match.group(1)

    # impl ... for Type
    impl_match = re.search(r'\bimpl\b[^\\n]*?\bfor\b\s*([^\s\{]+)', code)
    if impl_match:
        target = impl_match.group(1).strip()
        if '<' in target:
            target = target.split('<', 1)[0]
        if '::' in target:
            target = target.split('::')[-1]
        return target

    # use path::Type;
    use_matches = re.findall(r'\buse\s+[^;]+::(\w+)\s*;', code)
    if use_matches:
        return use_matches[0]

    return None


def _extract_renamed_from_doc(documentation: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract renamed from/to info from documentation.
    """
    if not documentation:
        return None, None
    match = re.search(r'renamed from\s+`([^`]+)`.*?to\s+`([^`]+)`', documentation, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _extract_deprecated_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    deprecated: API deprecation (Evolution-Aware View).
    
    Data: 30 samples, 100% have old_source_code and source_code.
    Strategy: use mapping to directly find replacement API.
    """
    api_name = sample.get('name', '')
    code = sample.get('code', '')
    
    if not code:
        return None
    
    # Extract imports
    imports = _extract_imports(code)
    
    # Prefer dataset replacement_api field (most reliable)
    # Otherwise dynamically extract from code
    replacement_api = sample.get('replacement_api', '') or _extract_replacement_from_code(api_name, code, sample)

    old_code = sample.get('old_source_code', '')
    new_code = sample.get('source_code', '')
    doc = sample.get('documentation', '')
    doc_old, doc_new = _extract_renamed_from_doc(doc)

    old_name = doc_old or _extract_declared_name(old_code) or api_name
    new_name = replacement_api or doc_new or _extract_declared_name(new_code) or api_name

    # Cannot determine old/new names; skip graph
    if not old_name or not new_name:
        return None
    
    # =========================================================================
    # Evolution-Aware View: deprecation sub-graph
    # Data: 30 samples, 100% with old_source_code and documentation
    # Core: documentation contains deprecation reason; old/new code diff is key
    # =========================================================================
    
    nodes = [
        {"id": "old", "type": "api", "name": old_name, "status": "deprecated", "role": "old"},
        {"id": "new", "type": "api", "name": new_name, "status": "active" if new_name != "REMOVED" else "none", "role": "new"}
    ]
    
    # Extract deprecation info from documentation (compact)
    deprecation_info = {"id": "deprecation", "type": "transition", "kind": "deprecation"}
    
    # Extract key info
    if doc:
        doc_lower = doc.lower()
        
        # Extract deprecation reason keywords
        if 'renamed' in doc_lower or 'rename' in doc_lower:
            deprecation_info['reason'] = 'renamed'
        elif 'moved' in doc_lower or 'restructure' in doc_lower:
            deprecation_info['reason'] = 'api_restructure'
        elif 'replaced' in doc_lower or 'superseded' in doc_lower:
            deprecation_info['reason'] = 'functionality_replaced'
        elif 'unsafe' in doc_lower or 'safety' in doc_lower:
            deprecation_info['reason'] = 'safety_concern'
        elif 'consistency' in doc_lower or 'naming' in doc_lower or 'directional' in doc_lower:
            deprecation_info['reason'] = 'naming_consistency'
        elif 'removed' in doc_lower or 'never returns' in doc_lower:
            deprecation_info['reason'] = 'removed_no_replacement'
        else:
            deprecation_info['reason'] = 'unspecified'
    
    # Analyse old/new code diff (migration complexity)
    if old_code and new_code:
        # Signature similarity
        old_sig_match = re.search(r'(pub\s+fn\s+\w+[^{]*)', old_code)
        new_sig_match = re.search(r'(pub\s+fn\s+\w+[^{]*)', new_code)
        
        if old_sig_match and new_sig_match:
            old_sig = old_sig_match.group(1)
            new_sig = new_sig_match.group(1)
            
            # Determine migration complexity
            if old_sig == new_sig:
                deprecation_info['migration_complexity'] = 'trivial_rename'
            elif old_name in new_sig or new_name in old_sig:
                deprecation_info['migration_complexity'] = 'simple_replacement'
            else:
                deprecation_info['migration_complexity'] = 'signature_changed'
    
    nodes.append(deprecation_info)
    
    edges = [
        {"from": "old", "to": "deprecation", "type": "deprecated_via"},
        {"from": "deprecation", "to": "new", "type": "recommends"},
        {"from": "old", "to": "new", "type": "replaced_by"}  # Explicit replacement relation
    ]
    
    # Add import dependencies
    if imports:
        nodes.append({"id": "code_imports", "type": "dependencies", "items": imports})
        edges.append({"from": "new", "to": "code_imports", "type": "requires"})
    
    result = {"nodes": nodes, "edges": edges}
    return filter_graph_nodes(result, category='api_evolution')


def _extract_stabilized_evolution(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    stabilized: Unstable->stable availability (Evolution-Aware View).
    """
    api_name = sample.get('name', '')
    from_ver = sample.get('from_version', '')
    to_ver = sample.get('to_version', '')
    code = sample.get('code', '')
    
    if not from_ver or not to_ver:
        return None
    
    # Extract imports
    imports = _extract_imports(code)
    
    # =========================================================================
    # Evolution-Aware View: stabilisation sub-graph
    # Core: version span is the most effective evolution signal
    # =========================================================================
    from_version = sample.get('from_version', '')
    to_version = sample.get('to_version', '')
    
    nodes = [
        {"id": "v0", "type": "api", "name": api_name, "status": "unstable", "version": from_version, "role": "old"},
        {"id": "v1", "type": "api", "name": api_name, "status": "stable", "version": to_version, "role": "new"},
    ]
    
    edges = [{
        "from": "v0", 
        "to": "v1", 
        "type": "stabilizes",
        "from_version": from_version,
        "to_version": to_version
    }]

    # Stabilisation change node (version info only)
    nodes.append({
        "id": "stab",
        "type": "change",
        "kind": "stabilization",
        "from_version": from_version,
        "to_version": to_version,
        "from_status": "unstable",
        "to_status": "stable",
    })
    edges.append({"from": "v0", "to": "stab", "type": "stabilizes_via"})
    edges.append({"from": "stab", "to": "v1", "type": "produces"})
    
    # Rare case with old_source_code (3/222); add code diff info
    old_code = sample.get('old_source_code', '')
    new_code = sample.get('source_code', '')
    if old_code and new_code and old_code != new_code:
        nodes.append({
            "id": "code_diff",
            "type": "code_change",
            "note": "rare_case_with_code_diff"
        })
        edges.append({"from": "v0", "to": "code_diff", "type": "modified_via"})
        edges.append({"from": "code_diff", "to": "v1", "type": "produces"})
    
    # Add imports node
    if imports:
        nodes.append({"id": "code_imports", "type": "dependencies", "items": imports})
        edges.append({"from": "v1", "to": "code_imports", "type": "requires"})
    
    result = {"nodes": nodes, "edges": edges}
    return filter_graph_nodes(result, category='api_evolution')


def _analyze_code_structure(code: str, api_name: str, source_lines: List[str]) -> Dict[str, Any]:
    """
    Extract Code-Context View nodes and edges from Reference Code y.
    
    Focus on API-call-centric information:
    1. Function definitions and signatures
    2. API call context (call line, arguments, return usage)
    3. Parameter mapping (func params -> API call args)
    """
    nodes = []
    edges = []
    # ========== 1. Extract function definitions ==========
    fn_spans = []
    fn_pattern = r'(pub(?:\([^)]+\))?\s+)?(async\s+)?(const\s+)?(unsafe\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^{;]+))?'
    for match in re.finditer(fn_pattern, code, re.MULTILINE):
        fn_name = match.group(5)
        params_str = match.group(6) or ''
        return_type = (match.group(7) or '').strip()
        
        # Extract context: locate function declaration line
        match_start = match.start()
        line_start = code.rfind('\n', 0, match_start) + 1
        line_end = code.find('\n', match_start)
        if line_end == -1:
            line_end = len(code)
        fn_context = code[line_start:line_end].strip()
        
        fn_id = f'fn_{fn_name}'
        nodes.append({
            'id': fn_id,
            'type': 'code_function',
            'name': fn_name,
            'params': params_str,
            'return_type': return_type,
            'context': fn_context
        })
        fn_spans.append((fn_id, match.end()))
        
        # Extract function params as nodes
        if params_str.strip():
            for i, raw in enumerate(_split_params(params_str)):
                param = raw.strip()
                if ':' in param:
                    param_name = param.split(':')[0].strip()
                    param_type = param.split(':', 1)[1].strip()
                    param_id = f'param_{fn_name}_{param_name}'
                    nodes.append({
                        'id': param_id,
                        'type': 'code_param',
                        'name': param_name,
                        'param_type': param_type,
                        'position': i,
                        'function': fn_name
                    })
                    edges.append({
                        'from': fn_id,
                        'to': param_id,
                        'type': 'has_param'
                    })
    
    # ========== 2. Extract API calls (Code-Context View) ==========
    # Match various API call patterns
    # Maintain seen set for O(1) dedup
    seen_node_ids = {n['id'] for n in nodes}
    
    api_patterns = [
        (rf'([\w.]+)\.{re.escape(api_name)}\s*\(', 'method_call'),  # receiver.api_name(...)
        (rf'([\w:]+)::{re.escape(api_name)}\s*\(', 'static_call'),  # Type/Trait::api_name(...)
        (rf'([\w:]+)::\s*<[^>]+>\s*::{re.escape(api_name)}\s*\(', 'static_call'),  # Type/Trait::<T>::api_name(...)
        (rf'([\w:]+)::{re.escape(api_name)}\s*::\s*<[^>]+>\s*\(', 'static_call'),  # Type/Trait::api_name::<T>(...)
        (rf'(?<![\w\.:]){re.escape(api_name)}\s*\(', 'direct_call'),  # api_name(...)
        (rf'\b\w+!\([^)]*{re.escape(api_name)}\s*\(', 'macro_call'),  # macro!(...api_name(...))
    ]
    
    call_counter = 0
    for pattern, call_type in api_patterns:
        for match in re.finditer(pattern, code, re.DOTALL):
            # Extract call-site line as context
            match_start = match.start()
            line_start = code.rfind('\n', 0, match_start) + 1
            line_end = code.find('\n', match_start)
            if line_end == -1:
                line_end = len(code)
            call_context = code[line_start:line_end].strip()
            
            receiver = match.group(1) if match.lastindex and match.lastindex >= 1 else None
            
            call_id = f'code_api_call_{call_counter}'
            call_counter += 1
            nodes.append({
                'id': call_id,
                'type': 'code_api_call',
                'api': api_name,
                'call_type': call_type,
                'receiver': receiver,
                'context': call_context
            })
            
            # Connect to API anchor node
            edges.append({
                'from': call_id,
                'to': 'api',
                'type': 'calls'
            })
            
            # Extract receiver info (content-based stable ID for dedup)
            if receiver:
                # Sanitise receiver string for stable ID
                safe_recv = re.sub(r'[^A-Za-z0-9_]+', '_', receiver)[:40]
                recv_id = f'receiver_{safe_recv}'
                # O(1) dedup via seen_node_ids set
                if recv_id not in seen_node_ids:
                    nodes.append({
                        'id': recv_id,
                        'type': 'code_receiver',
                        'name': receiver
                    })
                    seen_node_ids.add(recv_id)
                edges.append({
                    'from': call_id,
                    'to': recv_id,
                    'type': 'uses_receiver'
                })

            # Extract call arguments as arg nodes
            args_raw = _extract_balanced(code[match.end() - 1:], '(', ')')
            if args_raw is not None:
                for i, raw in enumerate(_split_params(args_raw)):
                    arg_text = raw.strip()
                    if not arg_text:
                        continue
                    arg_id = f'{call_id}_arg_{i}'
                    nodes.append({
                        'id': arg_id,
                        'type': 'code_call_arg',
                        'position': i,
                        'value': arg_text[:80],
                    })
                    edges.append({
                        'from': call_id,
                        'to': arg_id,
                        'type': 'has_arg'
                    })

    # fallback: synthesize a minimal call node when api_name appears but no call was detected
    if call_counter == 0 and api_name:
        # allow type/module/use or path mentions to create inferred calls
        appears_as_ident = re.search(rf'(?<![\\w:]){re.escape(api_name)}(?![\\w:])', code) is not None
        appears_in_use = re.search(rf'\\buse\\s+[^;]*\\b{re.escape(api_name)}\\b', code) is not None
        appears_as_type = re.search(rf'[:<\\s]\\s*{re.escape(api_name)}\\b', code) is not None
        appears_in_path = re.search(rf'\\b[\\w:]*{re.escape(api_name)}\\b', code) is not None
        if appears_as_ident or appears_in_use or appears_as_type or appears_in_path:
            call_id = 'code_api_call_0'
            nodes.append({
                'id': call_id,
                'type': 'code_api_call',
                'api': api_name,
                'call_type': 'inferred',
                'receiver': None,
                'context': None,
            })
            edges.append({
                'from': call_id,
                'to': 'api',
                'type': 'calls'
            })
    
    # ========== 4. Extract use imports (dependencies) ==========
    use_pattern = r'use\s+([\w:]+(?:::\{[^}]+\})?)'
    import_counter = 0
    for match in re.finditer(use_pattern, code):
        import_path = match.group(1)
        
        # Keep only imports containing the API
        if api_name in import_path or 'std' in import_path:
            import_id = f'import_{import_counter}'
            import_counter += 1
            nodes.append({
                'id': import_id,
                'type': 'code_import',
                'path': import_path
            })
    
    # ========== 5. Extract return statements (within functions) ==========
    ret_counter = 0
    for fn_id, start_idx in fn_spans:
        brace_start = code.find('{', start_idx)
        if brace_start == -1:
            continue
        body = _extract_balanced(code[brace_start:], '{', '}')
        if body is None:
            continue
        # Explicit return
        for m in re.finditer(r'\breturn\s+([^;]+);', body):
            ret_value = m.group(1).strip()
            if not ret_value:
                continue
            ret_id = f'return_{ret_counter}'
            ret_counter += 1
            nodes.append({
                'id': ret_id,
                'type': 'code_return',
                'value': ret_value[:50],
                'return_kind': 'explicit'
            })
            edges.append({'from': fn_id, 'to': ret_id, 'type': 'returns'})
            if api_name in ret_value:
                # Directly find API call node
                api_call_id = next((n['id'] for n in nodes if n.get('type') == 'code_api_call'), None)
                if api_call_id:
                    edges.append({'from': ret_id, 'to': api_call_id, 'type': 'returned_by'})
        # Implicit return (last expression)
        last_expr = None
        for line in reversed(body.splitlines()):
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
            if stripped.startswith('}'):
                continue
            if stripped.endswith(';') or stripped.startswith('return '):
                break
            if re.match(r'^(let|use|static|const)\b', stripped):
                break
            last_expr = stripped
            break
        if last_expr:
            ret_id = f'return_{ret_counter}'
            ret_counter += 1
            nodes.append({
                'id': ret_id,
                'type': 'code_return',
                'value': last_expr[:50],
                'return_kind': 'implicit'
            })
            edges.append({'from': fn_id, 'to': ret_id, 'type': 'returns'})
            if api_name in last_expr:
                # Optimised: use generator for direct lookup
                api_call_id = next((n['id'] for n in nodes if n.get('type') == 'code_api_call'), None)
                if api_call_id:
                    edges.append({'from': ret_id, 'to': api_call_id, 'type': 'returned_by'})
    
    return {'nodes': nodes, 'edges': edges}


def build_graph_entry(idx: int, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build heterogeneous API evolution graph for a single sample.
    
    Core approach:
    1. Build differentiated Evolution-Aware sub-graph by change_type
    2. Leverage old_source_code, source_code, replacement_api fields
    3. Fuse Evolution-Aware and Code-Context View information
    
    Output fields:
    - Data: function_signature, api, code, query, version, change_type
    - Code-Context: nodes (graph nodes), edges (graph edges)
    - Evolution: evolution (API evolution sub-graph)
    - Meta: version_info, evolution_type
    """
    code = sample.get('code', '').strip()
    api_name = sample.get('name', '').strip()
    change_type = sample.get('change_type', '')
    
    if not code or not api_name:
        return None
    
    source_lines = code.splitlines()
    
    # 1. Extract Code-Context View from Reference Code y
    code_structure = _analyze_code_structure(code, api_name, source_lines)
    
    # 2. Parse function_signature (for parameter name binding)
    func_sig_raw = sample.get('function_signature', '')
    func_sig_parsed = parse_function_signature(func_sig_raw)
    
    # =========================================================================
    # 3. Extract Evolution-Aware sub-graph
    # Build differentiated evolution by change_type
    # =========================================================================
    api_evolution = extract_api_evolution(sample)
    
    # Assemble final graph: merge all nodes and edges
    all_nodes = []
    all_edges = []
    
    # V_API anchor node (api_updated)
    api_node = {
        'id': 'api',
        'type': 'api_updated',
        'name': api_name,
        'module': sample.get('module', ''),
        'change_type': change_type,
        'from_version': sample.get('from_version', ''),
        'to_version': sample.get('to_version', ''),
    }
    # For deprecated type, add replacement_api
    if change_type == 'deprecated':
        replacement = sample.get('replacement_api', '')
        if replacement:
            api_node['replacement_api'] = replacement
    all_nodes.append(api_node)
    
    # Code-Context View nodes (V_code)
    if code_structure:
        for node in code_structure.get('nodes', []):
            node_type = node.get('type', '')
            # Keep core code structure nodes only
            if node_type.startswith('code_'):
                all_nodes.append(node)
        
        # Add Code-Context edges
        node_ids = {n['id'] for n in all_nodes}
        for edge in code_structure.get('edges', []):
            # Fix: calls (bridge) edges point to api anchor
            if edge.get('type') == 'calls':
                if edge.get('to') == 'api' and 'api' in node_ids:
                    all_edges.append(edge)
            elif edge.get('from') in node_ids and edge.get('to') in node_ids:
                all_edges.append(edge)
        for node in all_nodes:
            if node.get('type') == 'code_function':
                all_edges.append({
                    'from': node['id'],
                    'to': 'api',
                    'type': 'implements_query'
                })
    
    # Evolution-Aware View nodes and edges
    evo_old_id = None
    evo_new_id = None
    if api_evolution:
        evo_nodes = api_evolution.get('nodes', [])
        evo_edges = api_evolution.get('edges', [])
        
        # Prefix evolution node IDs to avoid conflicts
        evo_node_id_map = {}
        for node in evo_nodes:
            old_id = node.get('id', '')
            new_id = f"evo_{old_id}"
            evo_node_id_map[old_id] = new_id
            node['id'] = new_id
            all_nodes.append(node)
            if node.get('role') == 'old':
                evo_old_id = new_id
            elif node.get('role') == 'new':
                evo_new_id = new_id
        
        # Update edge ID references
        for edge in evo_edges:
            edge['from'] = evo_node_id_map.get(edge['from'], edge['from'])
            edge['to'] = evo_node_id_map.get(edge['to'], edge['to'])
            edge['type'] = edge.get('type') or edge.get('relation') or 'relates'
            all_edges.append(edge)
        
        # Connect Evolution-Aware nodes to API anchor
        # Connect evolution 'new' nodes to api anchor
        for node in all_nodes:
            if node.get('role') == 'new' and node['id'].startswith('evo_'):
                all_edges.append({
                    'from': node['id'],
                    'to': 'api',
                    'type': 'defines_current'
                })
            elif node.get('role') == 'old' and node['id'].startswith('evo_'):
                edge_type = 'deprecated_by'
                if change_type == 'stabilized':
                    edge_type = 'stabilized_from'
                elif change_type == 'implicit':
                    edge_type = 'previous_impl_of'
                elif change_type == 'signature':
                    edge_type = 'defines_previous'
                all_edges.append({
                    'from': node['id'],
                    'to': 'api',
                    'type': edge_type
                })
    
    # Source Structure sub-graph (signature type only, avoids library noise)
    old_source = sample.get('old_source_code', '').strip()
    new_source = sample.get('source_code', '').strip()
    
    if change_type == 'signature' and old_source and new_source and old_source != new_source:
        # Extract old source code structure
        old_source_structure = _extract_source_code_structure(old_source, api_name, 'old')
        new_source_structure = _extract_source_code_structure(new_source, api_name, 'new')
        
        # Build evolution edges before modifying node IDs
        _add_evolution_edges(old_source_structure, new_source_structure, all_edges)
        
        # Add old code nodes (with prefixed IDs)
        for node in old_source_structure.get('nodes', []):
            node['id'] = f"old_{node['id']}"
            node['source_role'] = 'old'
            all_nodes.append(node)
        
        # Add new code nodes (with prefixed IDs)
        for node in new_source_structure.get('nodes', []):
            node['id'] = f"new_{node['id']}"
            node['source_role'] = 'new'
            all_nodes.append(node)
        
        # Add edges (with updated ID references)
        node_ids = {n['id'] for n in all_nodes}
        for edge in old_source_structure.get('edges', []):
            edge['from'] = f"old_{edge['from']}"
            edge['to'] = f"old_{edge['to']}"
            if edge['from'] in node_ids and edge['to'] in node_ids:
                all_edges.append(edge)
        
        for edge in new_source_structure.get('edges', []):
            edge['from'] = f"new_{edge['from']}"
            edge['to'] = f"new_{edge['to']}"
            if edge['from'] in node_ids and edge['to'] in node_ids:
                all_edges.append(edge)
        
        # Connect old/new structure nodes to evo_old/evo_new
        if evo_old_id or evo_new_id:
            attach_types = {
                'function_def', 'method_call', 'impl_block',
                'struct_def', 'enum_def', 'trait_def', 'type_alias',
                'param', 'macro_call'
            }
            for node in all_nodes:
                ntype = node.get('type')
                if ntype not in attach_types:
                    continue
                if node.get('source_role') == 'old' and evo_old_id:
                    all_edges.append({
                        'from': evo_old_id,
                        'to': node['id'],
                        'type': 'contains'
                    })
                elif node.get('source_role') == 'new' and evo_new_id:
                    all_edges.append({
                        'from': evo_new_id,
                        'to': node['id'],
                        'type': 'contains'
                    })
        
    elif change_type == 'signature' and new_source:
        # Only new code available: add source_code structure
        new_source_structure = _extract_source_code_structure(new_source, api_name, 'current')
        for node in new_source_structure.get('nodes', []):
            node['id'] = f"src_{node['id']}"
            all_nodes.append(node)
        
        node_ids = {n['id'] for n in all_nodes}
        for edge in new_source_structure.get('edges', []):
            edge['from'] = f"src_{edge['from']}"
            edge['to'] = f"src_{edge['to']}"
            if edge['from'] in node_ids and edge['to'] in node_ids:
                all_edges.append(edge)
    
    # Parameter constraint nodes (func_param, func_return)
    if func_sig_parsed and isinstance(func_sig_parsed, dict):
        func_name = func_sig_parsed.get('name')
        func_node_id = f"fn_{func_name}" if func_name else None
        all_node_ids = {n.get('id') for n in all_nodes}
        if func_node_id not in all_node_ids:
            func_node_id = None
        code_param_names = {n.get('name') for n in all_nodes if n.get('type') == 'code_param'}
        for param in func_sig_parsed.get('params', [])[:3]:  # Keep top 3 params
            param_name = param.get('name', '')
            param_type = param.get('type', '')
            if param_name and param_type:
                if param_name in code_param_names:
                    continue
                param_node_id = f'param_{param_name}'
                all_nodes.append({
                    'id': param_node_id,
                    'type': 'func_param',
                    'name': param_name,
                    'param_type': param_type
                })
                # Connect to function node (param_of relation)
                if func_node_id:
                    all_edges.append({
                        'from': param_node_id,
                        'to': func_node_id,
                        'type': 'param_of'
                    })
        
        # Add return type node
        return_type = func_sig_parsed.get('return_type', '')
        if return_type:
            all_nodes.append({
                'id': 'return_type',
                'type': 'func_return',
                'return_type': return_type
            })
            if func_node_id:
                all_edges.append({
                    'from': func_node_id,
                    'to': 'return_type',
                    'type': 'returns'
                })

    # Bind signature constraints to params and API call args
    func_params = [n for n in all_nodes if n.get('type') == 'func_param']
    code_params = sorted(
        [n for n in all_nodes if n.get('type') == 'code_param'],
        key=lambda n: n.get('position', 0),
    )
    for i, fnode in enumerate(func_params):
        if i < len(code_params):
            all_edges.append({
                'from': fnode['id'],
                'to': code_params[i]['id'],
                'type': 'constrains_param'
            })

    call_nodes = sorted(
        [n for n in all_nodes if n.get('type') == 'code_api_call'],
        key=lambda n: str(n.get('id', ''))
    )
    if call_nodes:
        call_id = call_nodes[0].get('id')
        if call_id:
            arg_nodes = []
            for edge in all_edges:
                if edge.get('type') == 'has_arg' and edge.get('from') == call_id:
                    arg_nodes.append(edge.get('to'))
            arg_nodes = [
                n for n in all_nodes
                if n.get('id') in arg_nodes and n.get('type') == 'code_call_arg'
            ]
            arg_nodes.sort(key=lambda n: n.get('position', 0))
            for i, fnode in enumerate(func_params):
                if i < len(arg_nodes):
                    all_edges.append({
                        'from': fnode['id'],
                        'to': arg_nodes[i]['id'],
                        'type': 'constrains_arg'
                    })

    if any(n.get('id') == 'return_type' for n in all_nodes):
        for edge in list(all_edges):
            if edge.get('type') == 'returned_by':
                all_edges.append({
                    'from': 'return_type',
                    'to': edge.get('from'),
                    'type': 'constrains_return'
                })
    
    # Graph Pruning: 3-hop BFS from API call nodes
    api_call_ids = [n.get('id') for n in all_nodes if n.get('type') == 'code_api_call']
    all_nodes, all_edges = _slice_api_call_subgraph(all_nodes, all_edges, api_call_ids, hops=3)

    # Clean nodes: remove null/debug fields
    def clean_node(node):
        """Remove null values and debug fields."""
        cleaned = {}
        debug_fields = {'source', 'source_role'}
        for key, value in node.items():
            # Always keep id and type
            if key in ('id', 'type'):
                cleaned[key] = value
                continue
            # Skip debug fields
            if key in debug_fields:
                continue
            # Skip null/empty values
            if value in (None, '', [], {}):
                continue
            cleaned[key] = value
        return cleaned
    
    all_nodes = [clean_node(n) for n in all_nodes]
    
    # Build final result
    all_nodes.sort(key=lambda n: str(n.get('id', '')))
    all_edges.sort(key=lambda e: (str(e.get('from', '')), str(e.get('to', '')), str(e.get('type', ''))))
    result = {
        'id': idx,
        'signature': sample.get('signature', ''),
        'nodes': all_nodes,
        'edges': all_edges,
        'api': {
            'name': api_name,
            'module': sample.get('module', ''),
            'change_type': change_type,
            'from_version': sample.get('from_version', ''),
            'to_version': sample.get('to_version', ''),
        },
        'code': code,
    }
    
    # For deprecated type, add replacement_api
    if change_type == 'deprecated':
        replacement = sample.get('replacement_api', '')
        if replacement:
            result['replacement_api'] = replacement
    
    return result


def build_and_save_graphs(dataset, output_dir: str):
    """Build and save heterogeneous API evolution graphs for all samples."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    stats = {
        'total': len(dataset),
        'success': 0,
        'failed': 0,
        # Graph statistics
        'total_nodes': 0,
        'total_edges': 0,
        'node_types': {},
        'edge_types': {},
        # Evolution sub-graph statistics
        'change_type_stats': {
            'stabilized': {'count': 0, 'with_evolution': 0},
            'signature': {'count': 0, 'with_evolution': 0},
            'deprecated': {'count': 0, 'with_evolution': 0},
            'implicit': {'count': 0, 'with_evolution': 0},
        },
        'evolution_stats': {
            'has_old_source': 0,
            'has_source_diff': 0,
            'has_replacement_api': 0,
        },
    }
    
    print(f"\n{'='*60}")
    print(f"Building graphs: {len(dataset)} samples")
    print(f"Format: Heterogeneous API Evolution Graph")
    print(f"{'='*60}\n")
    
    start_time = perf_counter()
    
    for idx in tqdm(range(len(dataset)), desc="Building graphs"):
        sample = dataset[idx]
        entry = build_graph_entry(idx, sample)
        
        if entry:
            results.append(entry)
            stats['success'] += 1
            
            # Graph statistics
            num_nodes = len(entry.get('nodes', []))
            num_edges = len(entry.get('edges', []))
            
            stats['total_nodes'] += num_nodes
            stats['total_edges'] += num_edges
            
            # Node type statistics
            for node in entry.get('nodes', []):
                ntype = node.get('type', 'unknown')
                stats['node_types'][ntype] = stats['node_types'].get(ntype, 0) + 1
            
            # Edge type statistics
            for edge in entry.get('edges', []):
                etype = edge.get('type', 'unknown')
                stats['edge_types'][etype] = stats['edge_types'].get(etype, 0) + 1
            
            # Evolution sub-graph statistics
            change_type = entry.get('api', {}).get('change_type', '')
            if change_type in stats['change_type_stats']:
                stats['change_type_stats'][change_type]['count'] += 1
                # Check for evolution nodes (ID prefix evo_)
                if any(n.get('id', '').startswith('evo_') for n in entry.get('nodes', [])):
                    stats['change_type_stats'][change_type]['with_evolution'] += 1
            
            # Evolution coverage stats (based on raw sample fields)
            old_src = (sample.get('old_source_code') or '').strip()
            new_src = (sample.get('source_code') or '').strip()
            if old_src:
                stats['evolution_stats']['has_old_source'] += 1
            if old_src and new_src and old_src != new_src:
                stats['evolution_stats']['has_source_diff'] += 1
            if sample.get('replacement_api') or entry.get('replacement_api'):
                stats['evolution_stats']['has_replacement_api'] += 1
            
        else:
            stats['failed'] += 1
    
    build_time = perf_counter() - start_time
    
    # =========================================================================
    # Save graph data
    # =========================================================================
    
    output_file = output_path / "rustevo_graphs.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    stats['build_time'] = round(build_time, 2)
    stats['file_size_mb'] = round(output_file.stat().st_size / 1024 / 1024, 2)
    stats['avg_nodes_per_sample'] = round(stats['total_nodes'] / max(stats['success'], 1), 1)
    stats['avg_edges_per_sample'] = round(stats['total_edges'] / max(stats['success'], 1), 1)
    
    stats_file = output_path / "build_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")
    print(f"Built: {stats['success']}/{stats['total']}")
    print(f"File size: {stats['file_size_mb']} MB")
    print(f"Total nodes: {stats['total_nodes']} (avg {stats['avg_nodes_per_sample']}/sample)")
    print(f"Total edges: {stats['total_edges']} (avg {stats['avg_edges_per_sample']}/sample)")
    
    if stats['node_types']:
        print(f"\nNode type distribution:")
        total_nodes = stats['total_nodes']
        for ntype, count in sorted(stats['node_types'].items(), key=lambda x: -x[1])[:10]:
            pct = count / total_nodes * 100 if total_nodes > 0 else 0
            print(f"  {ntype}: {count} ({pct:.1f}%)")
    
    if stats['edge_types']:
        print(f"\nEdge type distribution:")
        total_edges = stats['total_edges']
        for etype, count in sorted(stats['edge_types'].items(), key=lambda x: -x[1])[:10]:
            pct = count / total_edges * 100 if total_edges > 0 else 0
            print(f"  {etype}: {count} ({pct:.1f}%)")
    
    # Print evolution sub-graph stats
    print(f"\n{'='*40}")
    print(f"Evolution-Aware sub-graph stats:")
    print(f"{'='*40}")
    for ct, ct_stats in stats['change_type_stats'].items():
        if ct_stats['count'] > 0:
            evo_rate = ct_stats['with_evolution'] / ct_stats['count'] * 100
            print(f"  {ct}: {ct_stats['count']} samples, {ct_stats['with_evolution']} with evolution ({evo_rate:.1f}%)")
    
    print(f"\nEvolution coverage:")
    print(f"  has old_source_code: {stats['evolution_stats']['has_old_source']}")
    print(f"  has source_diff: {stats['evolution_stats']['has_source_diff']}")
    print(f"  has replacement_api: {stats['evolution_stats']['has_replacement_api']}")
    
    print(f"\nBuild time: {build_time:.2f}s")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*60}\n")
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Build code graphs for RustEvo")
    parser.add_argument("--output_dir", type=str, default="./data/rustevo_graphs")
    parser.add_argument("--dataset_size_limit", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"RustEvo Graph Builder")
    print(f"{'='*60}")
    print(f"  Innovation: Structure extraction (signature parsing, api_usage)")
    print(f"{'='*60}")
    print(f"Output: {args.output_dir}")
    print(f"Limit: {args.dataset_size_limit or 'All'}")
    
    model_name = args.model_name.split("/")[-1] if "/" in args.model_name else args.model_name
    dataset = RustEvoDataset(DATA_DIR, model_name=model_name, size=args.dataset_size_limit)
    print(f"\nLoaded {len(dataset)} samples")
    
    build_and_save_graphs(dataset=dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
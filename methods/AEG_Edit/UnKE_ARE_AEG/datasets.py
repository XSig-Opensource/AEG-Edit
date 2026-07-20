"""
data - data

with:
1. Token extractstrategy
2. graphdataloadBuild
3. dataconfig
"""

import torch
import re
from typing import Dict, Set, Optional, List, Tuple, Any
from transformers import AutoTokenizer


def extract_critical_tokens_rustevo(
    tok: AutoTokenizer,
    answer: str,
    data: Dict,
    graph_data: Optional[Dict] = None,
    weight: float = 4.0,
) -> Dict[int, float]:
    """
    RustEvo: extract Rust API token
    
    ⚡ :API signature Coreinfo(function, parameter, type)
    strategy: API token,
    
    Note: find_critical_token_positions
    """
    if graph_data:
        api_info = graph_data.get("api", {})
        api_name = api_info.get("name", "")
        
        function_signature = ""
        for node in graph_data.get("nodes", []):
            if node.get("type") == "code_function":
                name = node.get("name", "")
                params = node.get("params", "")
                ret_type = node.get("return_type", "")
                function_signature = f"fn {name}({params}){' -> ' + ret_type if ret_type else ''}"
                break
    else:
        api_name = data.get("name", "")
        function_signature = data.get("function_signature", "")
    
    critical_positions = {}
    
    try:
        answer_tokens = tok(answer, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offsets = answer_tokens["offset_mapping"][0].tolist()
    except Exception:
        return {}
    
    def mark_text_positions(text_to_find: str):
        """ answer corresponding token"""
        if not text_to_find or len(text_to_find) < 2:
            return
        for match in re.finditer(re.escape(text_to_find), answer):
            span_start = match.start()
            span_end = match.end()
            for token_idx, (token_start, token_end) in enumerate(offsets):
                if token_start is None or token_end is None:
                    continue
                if not (token_end <= span_start or token_start >= span_end):
                    critical_positions[token_idx] = max(critical_positions.get(token_idx, 0), weight)
    
    if api_name:
        mark_text_positions(api_name)
        if '::' in api_name:
            short_name = api_name.split('::')[-1]
            mark_text_positions(short_name)
    
    if function_signature:
        fn_match = re.search(r'fn\s+(\w+)', function_signature)
        if fn_match:
            fn_name = fn_match.group(1)
            mark_text_positions(fn_name)
        
        param_matches = re.findall(r'(\w+)\s*:', function_signature)
        for param_name in param_matches:
            if len(param_name) > 2 and param_name not in ['fn', 'let', 'mut', 'pub', 'use', 'self']:
                mark_text_positions(param_name)
        
        type_matches = re.findall(r':\s*&?(?:mut\s+)?([A-Z]\w+)', function_signature)
        for type_name in type_matches:
            if len(type_name) > 1:
                mark_text_positions(type_name)
        
        ret_match = re.search(r'->\s*(.+?)(?:\s*where|\s*\{|;|$)', function_signature)
        if ret_match:
            ret_type = ret_match.group(1).strip()
            ret_types = re.findall(r'\b([A-Z]\w+)\b', ret_type)
            for rt in ret_types:
                if len(rt) > 1:
                    mark_text_positions(rt)
        
        generic_match = re.search(r'<([^>]+)>', function_signature)
        if generic_match:
            generics = generic_match.group(1)
            generic_params = re.findall(r'\b([A-Z]\w*)\b', generics)
            for gp in generic_params:
                if 1 <= len(gp) <= 20:
                    mark_text_positions(gp)
    
    return critical_positions


def load_rustevo_api_evolution_graph(
    graph_data: Optional[Dict],
    model,
    tok: AutoTokenizer,
    device: str = "cuda",
) -> Tuple[Any, Dict, torch.Tensor]:
    """
    RustEvo: Build API Evolution Graph(AST, data, call)
    
    completeversion - build_rustevo_dgl_graph
    
    Returns:
        g: DGL graphobject
        node_indices: nodeindexmap
        rel_emb: embedding
    """
    try:
        import dgl
    except ImportError:
        print("Warning: DGL not installed, graph features disabled")
        return None, {}, torch.tensor([])
    
    if graph_data is None or not graph_data:
        return None, {}, torch.tensor([])
    
    embed_layer = model.get_input_embeddings()
    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
    
    node_texts = {}  # node_id -> text
    node_types = {}  # node_id -> type
    edges = []  # [(src, dst, type), ...]
    
    for node in graph_data.get('nodes', []):
        nid = str(node.get('id', ''))
        if not nid:
            continue
        
        ntype = node.get('type', '')
        node_types[nid] = ntype
        
        if ntype == 'api_anchor':
            name = node.get('name', '')
            module = node.get('module', '')
            change_type = node.get('change_type', '')
            from_ver = node.get('from_version', '')
            to_ver = node.get('to_version', '')
            ver_range = f" {from_ver}→{to_ver}" if from_ver and to_ver else ""
            full_name = f"{module}::{name}" if module else name
            node_texts[nid] = f"⭐ANCHOR: {full_name} [{change_type}]{ver_range}"
        
        elif ntype == 'code_function':
            name = node.get('name', '')
            params = node.get('params', '')
            ret_type = node.get('return_type', '')
            visibility = node.get('visibility', '')
            generics = node.get('generics', '')
            context = node.get('context', '')
            vis = f"{visibility} " if visibility else ""
            gen = f"<{generics}>" if generics else ""
            ret = f" -> {ret_type}" if ret_type else ""
            signature = f"{vis}fn {name}{gen}({params}){ret}"
            if context:
                node_texts[nid] = f"{signature} {{{context[:100]}..."
            else:
                node_texts[nid] = signature
        
        elif ntype == 'code_param':
            name = node.get('name', '')
            param_type = node.get('param_type', '')
            function = node.get('function', '')
            node_texts[nid] = f"{function}::{name}: {param_type}" if function else f"{name}: {param_type}"
        
        elif ntype == 'code_api_call':
            api_called = node.get('api', '')
            call_type = node.get('call_type', '')
            receiver = node.get('receiver', '')
            context = node.get('context', '')
            if context:
                node_texts[nid] = context[:150]
            elif call_type == 'method_call' and receiver:
                node_texts[nid] = f"{receiver}.{api_called}(...)"
            else:
                node_texts[nid] = f"{api_called}(...)"
            
            if 'api' in node_texts:
                edges.append((nid, 'api', 'invokes'))

        elif ntype == 'code_call_arg':
            value = node.get('value', '')
            position = node.get('position', None)
            if position is not None:
                node_texts[nid] = f"arg{position}: {value}"
            else:
                node_texts[nid] = f"arg: {value}"
        
        elif ntype == 'code_receiver':
            name = node.get('name', '')
            node_texts[nid] = f"receiver: {name}"
        
        elif ntype == 'code_return':
            value = node.get('value', '')
            return_kind = node.get('return_kind', '')
            context = node.get('context', '')
            kind_prefix = f"[{return_kind}] " if return_kind else ""
            if context:
                node_texts[nid] = f"{kind_prefix}return {context[:100]}"
            else:
                node_texts[nid] = f"{kind_prefix}return {value}" if value else f"{kind_prefix}return"
        
        elif ntype == 'code_import':
            path = node.get('path', '')
            node_texts[nid] = f"use {path}"
        
        elif ntype == 'code_variable':
            name = node.get('name', '')
            var_type = node.get('var_type', '')
            init_value = node.get('init_value', '')
            is_mut = node.get('is_mut', False)
            mut_str = 'mut ' if is_mut else ''
            node_texts[nid] = f"let {mut_str}{name}: {var_type} = {init_value}" if var_type else f"let {mut_str}{name} = {init_value}"
        
        elif ntype == 'func_param':
            name = node.get('name', '')
            param_type = node.get('param_type', '')
            node_texts[nid] = f"param {name}: {param_type}"
        
        elif ntype == 'func_return':
            return_type = node.get('return_type', '')
            node_texts[nid] = f"returns {return_type}"
        
        elif ntype == 'function_def':
            name = node.get('name', '')
            return_type = node.get('return_type', '')
            role = node.get('role', '')
            node_texts[nid] = f"fn {name}() -> {return_type} [{role}]"
        
        elif ntype == 'param':
            name = node.get('name', '')
            param_type = node.get('param_type', '')
            role = node.get('role', '')
            node_texts[nid] = f"{name}: {param_type} [{role}]"
        
        elif ntype == 'method_call':
            name = node.get('name', '')
            role = node.get('role', '')
            node_texts[nid] = f".{name}() [{role}]"
        
        elif ntype in ['struct_def', 'enum_def', 'trait_def', 'type_alias']:
            name = node.get('name', '')
            role = node.get('role', '')
            node_texts[nid] = f"{ntype.replace('_', ' ')} {name} [{role}]"
        
        elif ntype == 'impl_block':
            target_type = node.get('target_type', '')
            trait_impl = node.get('trait_impl', '')
            role = node.get('role', '')
            if trait_impl:
                node_texts[nid] = f"impl {trait_impl} for {target_type} [{role}]"
            else:
                node_texts[nid] = f"impl {target_type} [{role}]"
        
        elif ntype == 'import_group':
            imports = node.get('imports', [])
            role = node.get('role', '')
            node_texts[nid] = f"imports: {', '.join(imports[:3])} [{role}]"
        
        elif ntype == 'macro_call':
            name = node.get('name', '')
            role = node.get('role', '')
            node_texts[nid] = f"{name}! [{role}]"
        
        elif ntype == 'api':
            name = node.get('name', '')
            status = node.get('status', '')
            version = node.get('version', '')
            role = node.get('role', '')
            if role in ['old', 'new']:
                role_tag = f" 【{role.upper()}】"
            else:
                role_tag = f" [{role}]" if role else ""
            ver_tag = f" @v{version}" if version else ""
            node_texts[nid] = f"api {name} ({status}){ver_tag}{role_tag}"
        
        elif ntype == 'sig':
            api_name = node.get('api', '')
            role = node.get('role', '')
            params = node.get('params', [])
            return_type = node.get('return_type', '')
            visibility = node.get('visibility', '')
            is_const = node.get('is_const', False)
            is_unsafe = node.get('is_unsafe', False)
            is_async = node.get('is_async', False)
            
            modifiers = []
            if visibility:
                modifiers.append(visibility)
            if is_const:
                modifiers.append('const')
            if is_unsafe:
                modifiers.append('unsafe')
            if is_async:
                modifiers.append('async')
            
            modifier_str = ' '.join(modifiers) + ' ' if modifiers else ''
            param_str = ', '.join([f"{p.get('name', '')}: {p.get('type', '')}" for p in params[:3]])
            if len(params) > 3:
                param_str += ', ...'
            ret_str = f" -> {return_type}" if return_type else ""
            role_tag = f" 【{role.upper()}】" if role in ['old', 'new'] else (f" [{role}]" if role else "")
            node_texts[nid] = f"sig {modifier_str}fn {api_name}({param_str}){ret_str}{role_tag}"
        
        elif ntype == 'impl':
            api_name = node.get('api', '')
            role = node.get('role', '')
            source_code = node.get('source_code', '')
            if source_code:
                code_snippet = source_code.split('\n')[0][:100] if '\n' in source_code else source_code[:100]
                role_tag = f" [{role}]" if role else ""
                node_texts[nid] = f"impl {api_name}{role_tag}: {code_snippet}"
            else:
                role_tag = f" [{role}]" if role else ""
                node_texts[nid] = f"impl {api_name}{role_tag}"
        
        elif ntype == 'change':
            kind = node.get('kind', '')
            from_ver = node.get('from_version', '')
            to_ver = node.get('to_version', '')
            from_status = node.get('from_status', '')
            to_status = node.get('to_status', '')
            old_value = node.get('old', '')
            new_value = node.get('new', '')
            param_info = node.get('param', '') or node.get('params', '')
            
            # Mark change nodes
            ver_info = f" v{from_ver}→v{to_ver}" if from_ver and to_ver else ""
            status_info = f" ({from_status}→{to_status})" if from_status and to_status else ""
            
            change_detail = ""
            if old_value and new_value:
                change_detail = f" [{old_value}→{new_value}]"
            elif param_info:
                change_detail = f" [param: {param_info}]"
            
            node_texts[nid] = f"CHANGE: {kind}{status_info}{ver_info}{change_detail}"
        
        elif ntype == 'transition':
            kind = node.get('kind', 'transition')
            reason = node.get('reason', '')
            migration = node.get('migration_complexity', '')
            is_breaking = node.get('is_breaking', False)
            breaking_tag = " [BREAKING]" if is_breaking else ""
            detail_parts = []
            if reason:
                detail_parts.append(f"reason: {reason[:100]}")
            if migration:
                detail_parts.append(f"migration: {migration}")
            detail = ', '.join(detail_parts)
            node_texts[nid] = f"{kind}{breaking_tag}: {detail}" if detail else f"{kind}{breaking_tag}"
        
        elif ntype == 'dependencies':
            items = node.get('items', [])
            node_texts[nid] = f"deps: {', '.join(items[:3])}"
        
        else:
            name = node.get('name', nid)
            node_texts[nid] = f"{ntype}: {name}"
    
    for edge in graph_data.get('edges', []):
        src = str(edge.get('from', ''))
        dst = str(edge.get('to', ''))
        etype = edge.get('type') or edge.get('relation') or 'relates'
        
        if src in node_texts and dst in node_texts:
            edges.append((src, dst, etype))

    if len(node_texts) < 2:
        return None, {}, torch.tensor([])
    
    node_list = list(node_texts.keys())
    node_indices = {nid: i for i, nid in enumerate(node_list)}
    num_nodes = len(node_list)
    
    node_embeddings = []
    try:
        all_texts = [node_texts[nid] for nid in node_list]
        all_tokens = tok(
            all_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            token_embs = embed_layer(all_tokens['input_ids'].to(device))
            if 'attention_mask' in all_tokens:
                mask = all_tokens['attention_mask'].to(device).unsqueeze(-1)
                embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                embs = token_embs.mean(dim=1)
            embs_norm = embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            embs = embs / embs_norm
        node_embeddings = list(embs)
    except Exception as e:
        for nid in node_list:
            text = node_texts[nid]
            try:
                tokens = tok(text, return_tensors='pt', truncation=True, max_length=128)
                with torch.no_grad():
                    token_embs = embed_layer(tokens['input_ids'].to(device))
                    emb = token_embs.mean(dim=1).squeeze(0)
                    emb = emb / emb.norm().clamp(min=1e-8)
            except:
                emb = torch.zeros(hidden_dim, device=device)
            node_embeddings.append(emb)
    
    src_ids, dst_ids = [], []
    edge_types = set()
    edge_type_list = []
    
    for src, dst, etype in edges:
        if src in node_indices and dst in node_indices:
            src_ids.append(node_indices[src])
            dst_ids.append(node_indices[dst])
            edge_types.add(etype)
            edge_type_list.append(etype)
    
    if not src_ids:
        return None, {}, torch.tensor([])
    
    edge_type_to_id = {et: i for i, et in enumerate(sorted(edge_types))}
    edge_type_ids = [edge_type_to_id[et] for et in edge_type_list]
    
    self_loop_type = len(edge_type_to_id)
    src_ids += list(range(num_nodes))
    dst_ids += list(range(num_nodes))
    edge_type_ids += [self_loop_type] * num_nodes
    
    import dgl
    g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes).to(device)
    g.ndata['feat'] = torch.stack(node_embeddings).to(device)
    g.ndata['id'] = torch.arange(num_nodes, device=device)
    g.edata['etype'] = torch.tensor(edge_type_ids, device=device)
    
    indegrees = g.in_degrees().float().clamp(min=1)
    g.ndata['norm'] = torch.pow(indegrees, -0.5).view(-1, 1)
    
    num_rels = len(edge_type_to_id) + 1
    rel_emb = torch.zeros(num_rels, hidden_dim, device=device)
    for et, idx in edge_type_to_id.items():
        try:
            et_text = et.replace('_', ' ')
            tokens = tok(et_text, return_tensors='pt', truncation=True, max_length=8)
            with torch.no_grad():
                emb = embed_layer(tokens['input_ids'].to(device)).mean(dim=1).squeeze(0)
                emb_norm = emb.norm()
                if emb_norm > 1e-8:
                    emb = emb / emb_norm
                rel_emb[idx] = emb
        except:
            pass
    node_type_map = {node_indices[nid]: node_types[nid] for nid in node_indices}
    node_role_map = {}
    for node_data in graph_data.get('nodes', []):
        nid = str(node_data.get('id', ''))
        if nid in node_indices:
            role = node_data.get('role', '')
            if role:
                node_role_map[node_indices[nid]] = role
    
    graph_meta = {
        'node_types': node_type_map,
        'node_roles': node_role_map,
        'num_api_nodes': sum(1 for nt in node_type_map.values() if nt in ['api_anchor', 'api']),
        'num_change_nodes': sum(1 for nt in node_type_map.values() if nt in ['change', 'transition']),
        'num_old_nodes': len([r for r in node_role_map.values() if r == 'old']),
        'num_new_nodes': len([r for r in node_role_map.values() if r == 'new']),
    }
    g.graph_meta = graph_meta
    node_type_map = {node_indices[nid]: node_types[nid] for nid in node_indices}
    node_role_map = {}
    for node_data in graph_data.get('nodes', []):
        nid = str(node_data.get('id', ''))
        if nid in node_indices:
            role = node_data.get('role', '')
            if role:
                node_role_map[node_indices[nid]] = role
    
    graph_meta = {
        'node_types': node_type_map,
        'node_roles': node_role_map,
        'num_api_nodes': sum(1 for nt in node_type_map.values() if nt in ['api_anchor', 'api']),
        'num_change_nodes': sum(1 for nt in node_type_map.values() if nt in ['change', 'transition']),
        'num_old_nodes': len([r for r in node_role_map.values() if r == 'old']),
        'num_new_nodes': len([r for r in node_role_map.values() if r == 'new']),
    }
    g.graph_meta = graph_meta
    
    return g, node_indices, rel_emb



def load_pyevo_graph(
    graph_data: Optional[Dict],
    model,
    tok: AutoTokenizer,
    device: str = "cuda",
) -> Tuple[Any, Dict, torch.Tensor]:
    """
    PyEvo: Build Python API heterogeneous graph

    Node types( RustEvo ):
      api_updated, code_function, code_param, func_param, code_api_call,
      code_call_arg (positional & keyword), func_return,
      change, new_signature, old_signature,
      api (role=old/new)

    Returns:
        g: DGL graphobject
        node_indices: nodeindexmap
        rel_emb: embedding
    """
    try:
        import dgl
    except ImportError:
        print("Warning: DGL not installed, graph features disabled")
        return None, {}, torch.tensor([])

    if graph_data is None or not graph_data:
        return None, {}, torch.tensor([])

    embed_layer = model.get_input_embeddings()
    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096

    node_texts = {}
    node_types = {}
    edges = []

    _param_nodes = {}  # node_id -> (name, annotation)
    for _n in graph_data.get('nodes', []):
        if _n.get('type') == 'code_param':
            _pid = str(_n.get('id', ''))
            if _pid:
                _param_nodes[_pid] = (_n.get('name', ''), _n.get('annotation', ''))
    _func_params_map = {}  # func_node_id -> [(name, annotation)]
    for _e in graph_data.get('edges', []):
        if _e.get('type') == 'has_param':
            _src = str(_e.get('from', ''))
            _dst = str(_e.get('to', ''))
            if _dst in _param_nodes:
                _func_params_map.setdefault(_src, []).append(_param_nodes[_dst])

    for node in graph_data.get('nodes', []):
        nid = str(node.get('id', ''))
        if not nid:
            continue
        ntype = node.get('type', '')
        node_types[nid] = ntype

        if ntype == 'api_updated':
            name = node.get('name', '')
            module = node.get('module', '')
            change_type = node.get('change_type', '')
            from_ver = node.get('from_version', '')
            to_ver = node.get('to_version', '')
            ver_range = f" v{from_ver}→{to_ver}" if from_ver and to_ver else ""
            full_path = name if '.' in name else (f"{module}.{name}" if module else name)
            node_texts[nid] = f"⭐ANCHOR: {full_path} [{change_type}]{ver_range}"

        elif ntype == 'code_function':
            name = node.get('name', '')
            ret_ann = node.get('return_annotation', '')
            params = _func_params_map.get(nid, [])
            param_str = ', '.join(f"{pn}: {pa}" if pa else pn for pn, pa in params)
            ret = f" -> {ret_ann}" if ret_ann else ""
            node_texts[nid] = f"def {name}({param_str}){ret}"

        elif ntype == 'code_param':
            name = node.get('name', '')
            ann = node.get('annotation', '')
            node_texts[nid] = f"{name}: {ann}" if ann else name

        elif ntype == 'func_param':
            name = node.get('name', '')
            ann = (node.get('annotation', '') or '')
            default = (node.get('default', '') or '')
            base = f"param {name}: {ann}" if ann else f"param {name}"
            node_texts[nid] = f"{base}={default}" if default else base

        elif ntype == 'code_api_call':
            api = node.get('api', '')
            api_short = api.split('.')[-1] if '.' in api else api
            call_repr = (node.get('call_repr', '') or '').strip()
            if call_repr:
                node_texts[nid] = call_repr
            else:
                node_texts[nid] = f"{api_short}(...)"

        elif ntype == 'code_call_arg':
            keyword = node.get('keyword', False)
            value = node.get('value', '')
            val = value if value else '?'
            if keyword:
                kname = node.get('name', '')
                node_texts[nid] = f"{kname}={val}"
            else:
                position = node.get('position', '')
                node_texts[nid] = f"arg[{position}]={val}"

        elif ntype == 'func_return':
            ann = node.get('annotation', '')
            node_texts[nid] = f"-> {ann}" if ann else "->"

        elif ntype == 'change':
            changes_list = node.get('changes', [])
            if changes_list:
                parts = []
                for c in changes_list:
                    kind = c.get('kind', '')
                    cname = c.get('name', '')
                    old_val = c.get('old_type', '') or c.get('old', '')
                    new_val = c.get('new_type', '') or c.get('new', '')
                    base = f"{kind}:{cname}" if cname else kind
                    if old_val and new_val:
                        base += f" [{old_val}→{new_val}]"
                    elif new_val:
                        base += f" [→{new_val}]"
                    elif old_val:
                        base += f" [{old_val}→]"
                    parts.append(base)
                if not parts:
                    continue
                node_texts[nid] = f"Δ {', '.join(parts)}"
            else:
                kind = node.get('kind', '')
                old_value = node.get('old', '')
                new_value = node.get('new', '')
                param_info = node.get('param', '') or node.get('params', '')
                change_detail = ''
                if old_value and new_value:
                    change_detail = f" [{old_value}→{new_value}]"
                elif param_info:
                    change_detail = f" [param: {param_info}]"
                if not kind and not change_detail:
                    continue
                node_texts[nid] = f"Δ {kind}{change_detail}"

        elif ntype in ('new_signature', 'old_signature'):
            role_tag = ' [NEW]' if ntype == 'new_signature' else ' [OLD]'
            fname = node.get('name', '') or (node.get('api', '').split('.')[-1])
            if not fname:
                continue
            params = node.get('params', [])
            ret_type = (node.get('return_type', '') or '')
            param_str = ', '.join(
                f"{p.get('name','')}: {(p.get('type','') or '')}" if p.get('type')
                else p.get('name', '')
                for p in params if p.get('name') not in ('self', 'cls')
            )
            ret_str = f" -> {ret_type}" if ret_type else ""
            node_texts[nid] = f"def {fname}({param_str}){ret_str}{role_tag}"

        elif ntype == 'api':
            api = node.get('api', '') or node.get('name', '')
            if not api:
                continue
            status = node.get('status', '')
            version = node.get('version', '')
            role = node.get('role', '')
            behavior_desc = (node.get('behavior_desc', '') or '').strip()
            role_tag = f" 【{role.upper()}】" if role in ('old', 'new') else (f" [{role}]" if role else "")
            ver_tag = f" @v{version}" if version else ""
            if behavior_desc:
                node_texts[nid] = f"api {api} ({status}){ver_tag}: {behavior_desc[:80]}{role_tag}"
            else:
                node_texts[nid] = f"api {api} ({status}){ver_tag}{role_tag}"

        else:
            name = node.get('name', '') or node.get('api', '') or nid
            node_texts[nid] = f"{ntype}: {name}"

    for edge in graph_data.get('edges', []):
        src = str(edge.get('from', ''))
        dst = str(edge.get('to', ''))
        etype = edge.get('type') or edge.get('relation') or 'relates'
        if src in node_texts and dst in node_texts:
            edges.append((src, dst, etype))

    _old_sig_id = next((nid for nid, nt in node_types.items() if nt == 'old_signature' and nid in node_texts), None)
    _new_sig_id = next((nid for nid, nt in node_types.items() if nt == 'new_signature' and nid in node_texts), None)
    if _old_sig_id and _new_sig_id:
        _has_direct = any(
            (s == _old_sig_id and d == _new_sig_id) or (s == _new_sig_id and d == _old_sig_id)
            for s, d, _ in edges
        )
        if not _has_direct:
            edges.append((_old_sig_id, _new_sig_id, 'evolved_to'))

    _BIDIR_ETYPES = {
        'evolves', 'evolves_to', 'deprecated_via', 'replaced_by', 'has_change', 'evolved_to',
    }
    edges.extend([
        (d, s, f'rev_{et}') for s, d, et in list(edges) if et in _BIDIR_ETYPES
    ])

    if len(node_texts) < 2:
        return None, {}, torch.tensor([])

    node_list = list(node_texts.keys())
    node_indices = {nid: i for i, nid in enumerate(node_list)}
    if 'api' not in node_indices:
        for _nid, _ntype in node_types.items():
            if _ntype == 'api_updated' and _nid in node_indices:
                node_indices['api'] = node_indices[_nid]
                break
    num_nodes = len(node_list)

    node_embeddings = []
    all_texts = [node_texts[nid] for nid in node_list]
    _max_chars = max(len(t) for t in all_texts)
    _tok_max_len = min(256, max(128, _max_chars // 3 + 16))
    try:
        all_tokens = tok(all_texts, return_tensors='pt', padding=True, truncation=True, max_length=_tok_max_len)
        with torch.no_grad():
            token_embs = embed_layer(all_tokens['input_ids'].to(device))
            if 'attention_mask' in all_tokens:
                mask = all_tokens['attention_mask'].to(device).unsqueeze(-1)
                embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                embs = token_embs.mean(dim=1)
            embs = embs / embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        node_embeddings = list(embs)
    except Exception:
        for nid in node_list:
            try:
                tokens = tok(node_texts[nid], return_tensors='pt', truncation=True, max_length=128)
                with torch.no_grad():
                    emb = embed_layer(tokens['input_ids'].to(device)).mean(dim=1).squeeze(0)
                    emb = emb / emb.norm().clamp(min=1e-8)
            except Exception:
                emb = torch.zeros(hidden_dim, device=device)
            node_embeddings.append(emb)

    src_ids, dst_ids = [], []
    edge_types = set()
    edge_type_list = []
    for src, dst, etype in edges:
        if src in node_indices and dst in node_indices:
            src_ids.append(node_indices[src])
            dst_ids.append(node_indices[dst])
            edge_types.add(etype)
            edge_type_list.append(etype)

    if not src_ids:
        return None, {}, torch.tensor([])

    edge_type_to_id = {et: i for i, et in enumerate(sorted(edge_types))}
    edge_type_ids = [edge_type_to_id[et] for et in edge_type_list]

    self_loop_type = len(edge_type_to_id)
    src_ids += list(range(num_nodes))
    dst_ids += list(range(num_nodes))
    edge_type_ids += [self_loop_type] * num_nodes

    g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes).to(device)
    g.ndata['feat'] = torch.stack(node_embeddings).to(device)
    g.ndata['id'] = torch.arange(num_nodes, device=device)
    g.edata['etype'] = torch.tensor(edge_type_ids, device=device)

    indegrees = g.in_degrees().float().clamp(min=1)
    g.ndata['norm'] = torch.pow(indegrees, -0.5).view(-1, 1)

    num_rels = len(edge_type_to_id) + 1
    rel_emb = torch.zeros(num_rels, hidden_dim, device=device)
    for et, idx in edge_type_to_id.items():
        try:
            tokens = tok(et.replace('_', ' '), return_tensors='pt', truncation=True, max_length=8)
            with torch.no_grad():
                emb = embed_layer(tokens['input_ids'].to(device)).mean(dim=1).squeeze(0)
                emb_norm = emb.norm()
                if emb_norm > 1e-8:
                    rel_emb[idx] = emb / emb_norm
        except Exception:
            pass

    node_type_map = {node_indices[nid]: node_types[nid] for nid in node_indices}
    node_role_map = {}
    for node_data in graph_data.get('nodes', []):
        nid = str(node_data.get('id', ''))
        if nid in node_indices:
            role = node_data.get('role', '')
            if role:
                node_role_map[node_indices[nid]] = role

    g.graph_meta = {
        'node_types': node_type_map,
        'node_roles': node_role_map,
        'num_api_nodes':      sum(1 for nt in node_type_map.values() if nt == 'api_updated'),
        'num_call_nodes':     sum(1 for nt in node_type_map.values() if nt == 'code_api_call'),
        'num_function_nodes': sum(1 for nt in node_type_map.values() if nt == 'code_function'),
        'num_evo_nodes':      sum(1 for i, nt in node_type_map.items()
                                  if nt in ('old_signature', 'new_signature')
                                  or (nt == 'api' and node_role_map.get(i) in ('old', 'new'))),
        'change_type':        graph_data.get('api', {}).get('change_type', ''),
    }

    return g, node_indices, rel_emb


def extract_critical_tokens_pyevo(
    tok: AutoTokenizer,
    answer: str,
    data: Dict,
    graph_data: Optional[Dict] = None,
    weight: float = 4.0,
) -> Dict[int, float]:
    """
    PyEvo: extract Python API token

    RustEvo : weight, Use.
      api_name/module > full_api_path > function_name > func_params
      > changed_params > replacement_api > signature > new_signature_params
    """
    _SKIP = {'def', 'return', 'None', 'True', 'False', 'self', 'cls',
             'int', 'str', 'bool', 'float', 'bytes', 'object', 'type'}

    if graph_data:
        api_info    = graph_data.get("api", {})
        api_name    = api_info.get("name", "")
        api_module  = api_info.get("module", "")
        api_signature = graph_data.get("signature", "")

        function_name       = ""
        code_func_ret_ann   = ""
        func_ret_ann        = ""
        func_params         = []
        changed_params          = []
        changed_param_new_types = []
        new_sig_fname       = ""
        new_sig_params      = []
        new_sig_ret         = ""
        old_sig_fname       = ""
        old_sig_params      = []
        replacement_api     = ""   # api(status=replacement)
        _new_api_fallback   = ""
        stable_api_name     = ""   # api(status=stable)
        new_behavior_api    = ""   # api(status=behavior, role=new)

        for node in graph_data.get("nodes", []):
            ntype = node.get("type", "")
            if ntype == "code_function" and not function_name:
                function_name = node.get("name", "")
                code_func_ret_ann = node.get("return_annotation", "")
            elif ntype == "func_param":
                pname = node.get("name", "")
                if pname and pname not in ("self", "cls"):
                    func_params.append((pname, node.get("annotation", "")))
            elif ntype == "func_return":
                func_ret_ann = node.get("annotation", "")
            elif ntype == "change":
                for c in node.get("changes", []):
                    pname = c.get("name", "")
                    if pname:
                        changed_params.append(pname)
                    new_t = c.get("new_type", "") or c.get("new", "")
                    if new_t and len(new_t) > 1:
                        changed_param_new_types.append(new_t)
            elif ntype == "new_signature":
                new_sig_fname = node.get("name", "")
                new_sig_params = node.get("params", [])
                new_sig_ret = node.get("return_type", "")
            elif ntype == "old_signature":
                old_sig_fname = node.get("name", "")
                old_sig_params = node.get("params", [])
            elif ntype == "api":
                role = node.get("role", "")
                status = node.get("status", "")
                api_val = node.get("api", "") or node.get("name", "")
                if role == "new" and api_val:
                    if status == "replacement" and not replacement_api:
                        replacement_api = api_val
                    elif status == "stable" and not stable_api_name:
                        stable_api_name = api_val
                    elif status == "behavior" and not new_behavior_api:
                        new_behavior_api = api_val
                    elif not _new_api_fallback:
                        _new_api_fallback = api_val

        if not func_ret_ann:
            func_ret_ann = code_func_ret_ann
        if not replacement_api and _new_api_fallback and _new_api_fallback != api_name:
            replacement_api = _new_api_fallback
    else:
        api_name          = data.get("name", "")
        api_module        = data.get("module", "")
        api_signature     = data.get("signature", "")
        function_name     = ""
        func_params       = []
        func_ret_ann      = ""
        changed_params          = []
        changed_param_new_types = []
        new_sig_fname = old_sig_fname = ""
        new_sig_params = old_sig_params = []
        new_sig_ret       = ""
        replacement_api   = ""
        stable_api_name   = ""
        new_behavior_api  = ""

    critical_positions = {}

    try:
        answer_tokens = tok(answer, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offsets = answer_tokens["offset_mapping"][0].tolist()
    except Exception:
        return {}

    def mark(text: str, w: float = weight):
        if not text or len(text) < 2:
            return
        for match in re.finditer(re.escape(text.strip()), answer):
            span_start, span_end = match.start(), match.end()
            for token_idx, (ts, te) in enumerate(offsets):
                if ts is None or te is None:
                    continue
                if not (te <= span_start or ts >= span_end):
                    critical_positions[token_idx] = max(critical_positions.get(token_idx, 0), w)

    if api_name:
        short_name = api_name.split(".")[-1] if "." in api_name else api_name
        mark(short_name)
    if api_module and len(api_module) > 2:
        mark(api_module)

    if api_module and api_name:
        short_name = api_name.split(".")[-1] if "." in api_name else api_name
        full_api_path = f"{api_module}.{short_name}"
        if len(full_api_path) > 3:
            mark(full_api_path)

    if function_name and len(function_name) > 2 and function_name not in _SKIP:
        mark(function_name)

    for pname, pann in func_params:
        if pname not in _SKIP and len(pname) > 1:
            mark(pname)
        if pann:
            for tname in re.findall(r'\b([A-Za-z]\w+)\b', pann):
                if tname not in _SKIP and len(tname) > 2:
                    mark(tname)

    if func_ret_ann:
        for tname in re.findall(r'\b([A-Za-z]\w+)\b', func_ret_ann):
            if tname not in _SKIP and len(tname) > 1:
                mark(tname)

    for pname in changed_params:
        if pname not in _SKIP and len(pname) > 1:
            mark(pname)

    for new_t in changed_param_new_types:
        for tname in re.findall(r'\b([A-Za-z]\w+)\b', new_t):
            if tname not in _SKIP and len(tname) > 2:
                mark(tname)

    if replacement_api:
        short = replacement_api.split(".")[-1] if "." in replacement_api else replacement_api
        if short not in _SKIP and len(short) > 2:
            mark(short)
        if "." in replacement_api and len(replacement_api) > 3:
            mark(replacement_api)

    if stable_api_name:
        short = stable_api_name.split(".")[-1] if "." in stable_api_name else stable_api_name
        if short not in _SKIP and len(short) > 2:
            mark(short)

    if new_behavior_api:
        short = new_behavior_api.split(".")[-1] if "." in new_behavior_api else new_behavior_api
        if short not in _SKIP and len(short) > 2:
            mark(short)

    if api_signature:
        for pname in re.findall(r'\b(\w+)\s*[=:,\)]', api_signature):
            if pname not in _SKIP and len(pname) > 2:
                mark(pname)
        ret_match = re.search(r'->\s*(.+?)$', api_signature)
        if ret_match:
            for tname in re.findall(r'\b([A-Z]\w+)\b', ret_match.group(1)):
                if len(tname) > 1:
                    mark(tname)

    for fname in [new_sig_fname, old_sig_fname]:
        if fname and fname not in _SKIP and len(fname) > 2:
            mark(fname)
    for p in new_sig_params:
        pname = p.get('name', '')
        ptype = p.get('type', '')
        if pname and pname not in ('self', 'cls') and pname not in _SKIP and len(pname) > 1:
            mark(pname)
        if ptype:
            for tname in re.findall(r'\b([A-Za-z]\w+)\b', ptype):
                if tname not in _SKIP and len(tname) > 2:
                    mark(tname)
    if new_sig_ret:
        for tname in re.findall(r'\b([A-Za-z]\w+)\b', new_sig_ret):
            if tname not in _SKIP and len(tname) > 2:
                mark(tname)

    return critical_positions



def extract_critical_tokens(
    tok: AutoTokenizer,
    answer: str,
    data: Dict,
    dataset_type: str,
    graph_data: Optional[Dict] = None,
    weight: float = 4.0,
) -> Dict[int, float]:
    """
    token extract
    
    Args:
        tok: Tokenizer
        answer:
        data: datasample
        dataset_type: Dataset type ("rustevo" or "pyevo")
        graph_data: graphdata()
        weight: token weights
    
    Returns:
        critical_positions: {position: weight}
    """
    if dataset_type == "rustevo":
        return extract_critical_tokens_rustevo(
            tok, answer, data, graph_data, weight
        )
    elif dataset_type == "pyevo":
        return extract_critical_tokens_pyevo(
            tok, answer, data, graph_data, weight
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def load_graph_data(
    graph_data: Optional[Dict],
    dataset_type: str,
    model,
    tok: AutoTokenizer,
    device: str = "cuda",
) -> Tuple[Any, Dict, torch.Tensor]:
    """
    graphdataload
    
    Args:
        graph_data: graphdata
        dataset_type: datatype
        model: model
        tok: Tokenizer
        device:
    
    Returns:
        g: DGL graphobject
        node_indices: nodeindexmap
        rel_emb: embedding
    """
    if dataset_type == "rustevo":
        return load_rustevo_api_evolution_graph(graph_data, model, tok, device)
    elif dataset_type == "pyevo":
        return load_pyevo_graph(graph_data, model, tok, device)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

"""
Construct AEG-Edit target vectors for AlphaEdit-ARE.

This module keeps the ARE sliding-window editing workflow, then injects
API Evolution Graph signals into API-relevant windows during target construction.
"""

from typing import Dict, List, Tuple, Optional, Set
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AlphaEdit_ARE_AEG.AlphaEdit_ARE_AEG_hparams import AlphaEditAREAEGHyperParams
except ImportError:
    from .AlphaEdit_ARE_AEG_hparams import AlphaEditAREAEGHyperParams

from util import nethook

try:
    from AlphaEdit_ARE_AEG.datasets import extract_critical_tokens, load_graph_data
except ImportError:
    from .datasets import extract_critical_tokens, load_graph_data

try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    print("Warning: DGL not available, AEG-Edit graph enhancement disabled")


def build_api_weighted_windows(
    critical_positions: Set[int],
    total_len: int,
    window_size: int,
    overlap: int = 0,
) -> List[Tuple[int, int, int]]:
    """
    API-weighted windowing used during AEG-Edit target construction.
    
    Args:
        critical_positions: token
        total_len:
        window_size:
        overlap: token(0)
    
    Returns:
        : [(start, end, hit_count), ...]
    """
    windows = []
    start = 0
    
    while start < total_len:
        end = min(total_len, start + window_size)
        
        hit_count = sum(1 for pos in critical_positions if start <= pos < end)
        
        windows.append((start, end, hit_count))
        
        if end == total_len:
            break
        
        start += window_size - overlap
    
    return windows


# Backward-compatible alias kept for existing experiment entrypoints.
build_api_edit_windows = build_api_weighted_windows


class ApiEvolutionGraphRGCNLayer(nn.Module):
    """
    Typed RGCN layer for API Evolution Graph message passing.
    
    : msg = (node_feat + relation_emb) @ W
    
    :
    1. Weight matrix: weight_neighbor, loop_weight, evolve_loop_weight
    2. RReLU function
    3. edge/edgenodeprocess
    """
    def __init__(self, in_feat, out_feat, num_rels, dropout=0.1, self_loop=True):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.self_loop = self_loop
        
        self.rel_proj = nn.Linear(in_feat, in_feat, bias=False)
        nn.init.eye_(self.rel_proj.weight)
        
        self.weight_neighbor = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        
        self.attn_src = nn.Parameter(torch.Tensor(1, out_feat))
        self.attn_dst = nn.Parameter(torch.Tensor(1, out_feat))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = nn.RReLU()
    
    def forward(self, g, h, rel_emb):
        """
        Args:
            g: DGL graph
            h: nodefeature [num_nodes, in_feat]
            rel_emb: embedding [num_rels, in_feat]
        """
        g.ndata['h'] = h
        g.ndata['h_trans'] = h @ self.weight_neighbor
        
        def msg_func(edges):
            relation = rel_emb.index_select(0, edges.data['etype'])
            if relation.shape[-1] != self.in_feat:
                relation = self.rel_proj(relation)
            relation = relation.view(-1, self.in_feat)
            node = edges.src['h'].view(-1, self.in_feat)
            
            msg = (node + relation) @ self.weight_neighbor  # [num_edges, out_feat]
            
            # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            msg_src = (msg * self.attn_src).sum(dim=-1, keepdim=True)  # [num_edges, 1]
            msg_dst = (edges.dst['h_trans'] * self.attn_dst).sum(dim=-1, keepdim=True)  # [num_edges, 1]
            attn_score = self.leaky_relu(msg_src + msg_dst)
            
            return {'msg': msg, 'attn': attn_score}
        
        def reduce_func(nodes):
            attn = torch.softmax(nodes.mailbox['attn'], dim=1)  # [num_nodes, num_neighbors, 1]
            h_neigh = (nodes.mailbox['msg'] * attn).sum(dim=1)  # [num_nodes, out_feat]
            return {'h_neigh': h_neigh}
        
        def apply_func(nodes):
            return {'h': nodes.data['h_neigh'] * nodes.data['norm']}
        
        g.update_all(msg_func, reduce_func, apply_func)
        node_repr = g.ndata['h']
        
        if self.self_loop:
            in_degrees = g.in_degrees().float()
            original_in_degrees = in_degrees - 1
            
            has_in_edges = (original_in_degrees > 0).float().unsqueeze(-1)
            no_in_edges = 1.0 - has_in_edges
            
            loop_message_normal = torch.mm(h, self.loop_weight)
            loop_message_evolve = torch.mm(h, self.evolve_loop_weight)
            
            loop_message = has_in_edges * loop_message_normal + no_in_edges * loop_message_evolve
            node_repr = node_repr + loop_message
        
        node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        
        return node_repr


class AEGEncoder(nn.Module):
    """
    Encode API Evolution Graphs into a hidden-space structural direction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        num_layers: int = 2,
        num_rels: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.rel_init_proj = nn.Linear(input_dim, hidden_dim)
        
        self.rel_emb = nn.Parameter(torch.Tensor(num_rels, hidden_dim))
        nn.init.xavier_uniform_(self.rel_emb)
        
        self.rgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.rgcn_layers.append(
                ApiEvolutionGraphRGCNLayer(hidden_dim, hidden_dim, num_rels, dropout=dropout, self_loop=True)
            )
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        g: 'dgl.DGLGraph',
        node_feats: torch.Tensor,
        init_rel_emb: torch.Tensor = None,
        target_node_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            g: DGL graph (with g.ndata['norm'])
            node_feats: nodefeature [num_nodes, input_dim]
            init_rel_emb: embedding()
            target_node_idx: targetnodeindex
            
        Returns:
            delta_feature: feature [input_dim]
        """
        h = self.input_proj(node_feats)  # [num_nodes, hidden_dim]
        
        if init_rel_emb is not None and init_rel_emb.numel() > 0:
            init_rel_projected = self.rel_init_proj(init_rel_emb)
            k = min(init_rel_projected.shape[0], self.rel_emb.shape[0])
            rel_emb = init_rel_projected[:k] + 0.1 * self.rel_emb[:k]
            if self.rel_emb.shape[0] > k:
                rel_emb = torch.cat([rel_emb, self.rel_emb[k:]], dim=0)
        else:
            rel_emb = self.rel_emb
        
        for layer in self.rgcn_layers:
            h = layer(g, h, rel_emb)
        
        h_target = h[target_node_idx]
        
        delta = self.output_proj(h_target) * self.scale
        
        self._last_delta = delta
        
        return delta


class GraphAlignmentAdapter(nn.Module):
    """
    GNN-LLM Adapter: GNN Output LLM
    
    :
    1. GNN Outputgraph, LLM token-level
    2. Adapter Use bottleneck structuredimension
    3. , trainingstable
    4. Use
    
    structure: Input -> Down-projection -> Activation -> Up-projection -> Gate -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
        
        self.input_ln = nn.LayerNorm(input_dim)
        
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, input_dim))
        else:
            # Down-projection
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            # Up-projection
            self.layers.append(nn.Linear(hidden_dim, input_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        self.output_ln = nn.LayerNorm(input_dim)
        
        self.gate = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """weights, adapter """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: GNN Output [input_dim] [batch, input_dim]
        
        Returns:
            adapted: Output [input_dim] [batch, input_dim]
        """
        is_1d = x.dim() == 1
        if is_1d:
            x = x.unsqueeze(0)
        
        residual = x
        
        h = self.input_ln(x)
        
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
                if self.dropout is not None:
                    h = self.dropout(h)
        
        h = self.output_ln(h)
        
        gate_value = torch.sigmoid(self.gate)
        
        output = residual + gate_value * h
        
        if is_1d:
            output = output.squeeze(0)
        
        return output
    
    def get_gate_value(self) -> float:
        """get"""
        return torch.sigmoid(self.gate).item()


# Public names used by the AEG-Edit release.
ApiEvolutionGraphEncoder = AEGEncoder
GraphToHiddenAlignmentAdapter = GraphAlignmentAdapter



def compute_aeg_edit_targets(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: AlphaEditAREAEGHyperParams,
    graph_data: Optional[Dict] = None,
    dataset_type: str = "rustevo",
) -> Tuple[List[int], List[torch.Tensor]]:
    """
    Compute graph-enhanced AEG-Edit targets on top of the ARE workflow.
    
    Supportdata:
    - rustevo: Rust API evolution ()
    - pyevo: Python API evolution (PyEvo)

    
    workflow:
    1. Build DGL graph(graphdata)
    2. graph-to-hidden alignment encoding
    3. API-weighted windowing
    4. : delta GNN parameter
    5. GNN Output delta
    """
    
    def log(msg: str):
        print(msg)
    
    def train_log(msg: str):
        print(msg)
    
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    #=2. Tokenize=
    target_ids = tok(data["answer"], return_tensors="pt").to("cuda")["input_ids"][0]
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    
    input_tok = tok(
        [data["question"]],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    token_weight = getattr(hparams, 'critical_token_weight', 4.0)
    
    critical_positions = extract_critical_tokens(
        tok=tok,
        answer=data["answer"],
        data=data,
        dataset_type=dataset_type,
        graph_data=graph_data,
        weight=token_weight,
    )
    
    token_weights = torch.ones(len(target_ids), device="cuda")
    for pos, weight in critical_positions.items():
        if pos < len(token_weights):
            token_weights[pos] = weight
    
    token_weights = token_weights / token_weights.mean()
    
    if len(critical_positions) > 0:
        log(f"Found {len(critical_positions)} critical tokens (weight: {token_weight})")

    if getattr(hparams, 'ablate_uniform_window', False):
        critical_positions = {}
        token_weights = torch.ones(len(target_ids), device="cuda")
        log("[Ablation] ablate_uniform_window=True: uniform token weights, all windows use full training")

    focus_windows = build_api_weighted_windows(
        critical_positions=set(critical_positions.keys()),
        total_len=len(target_ids),
        window_size=hparams.window_size,
        overlap=getattr(hparams, 'overlap', 0),
    )
    log(f"Generated {len(focus_windows)} API-weighted windows from critical positions")

    g, node_indices, init_rel_emb = None, {}, torch.tensor([])
    gnn_model = None
    adapter_model = None  # GNN-LLM Adapter
    
    if HAS_DGL and hparams.use_gnn and graph_data is not None:
        try:
            g, node_indices, init_rel_emb = load_graph_data(
                graph_data=graph_data,
                dataset_type=dataset_type,
                model=model,
                tok=tok,
                device="cuda"
            )
            
            if g is not None and g.num_nodes() >= 2:
                if hasattr(model.config, 'hidden_size'):
                    hidden_dim = model.config.hidden_size
                elif hasattr(model.config, 'n_embd'):
                    hidden_dim = model.config.n_embd
                else:
                    hidden_dim = 4096
                
                num_rels = init_rel_emb.shape[0] if init_rel_emb.numel() > 0 else 20
                
                gnn_model = AEGEncoder(
                    input_dim=hidden_dim,
                    hidden_dim=hparams.gnn_hidden_dim,
                    num_layers=hparams.gnn_num_layers,
                    num_rels=num_rels,
                    dropout=hparams.gnn_dropout,
                ).cuda()
                
                if init_rel_emb.numel() > 0:
                    with torch.no_grad():
                        projected_init = gnn_model.rel_init_proj(init_rel_emb)
                        k = min(projected_init.shape[0], gnn_model.rel_emb.shape[0])
                        gnn_model.rel_emb[:k].copy_(projected_init[:k])
                    log(f"Initialized {k} relation embeddings from graph")
                
                use_adapter = getattr(hparams, 'use_adapter', False)
                if use_adapter:
                    adapter_model = GraphAlignmentAdapter(
                        input_dim=hidden_dim,
                        hidden_dim=getattr(hparams, 'adapter_hidden_dim', 1024),
                        num_layers=getattr(hparams, 'adapter_num_layers', 2),
                        activation=getattr(hparams, 'adapter_activation', 'gelu'),
                        dropout=getattr(hparams, 'adapter_dropout', 0.1),
                    ).cuda()
                    
                    adapter_params = sum(p.numel() for p in adapter_model.parameters())
                    log(f"Adapter initialized: {adapter_params:,} parameters")
                else:
                    log("Adapter disabled")
                
                gnn_params = sum(p.numel() for p in gnn_model.parameters())
            else:
                g = None
        except Exception as e:
            import traceback
            print(f"[AEG] GNN init failed: {e}")
            traceback.print_exc()
            g = None

    base_input_ids = input_tok['input_ids']
    all_target = []
    all_idxs = []
    
    gnn_initial_state = None
    adapter_initial_state = None
    
    if gnn_model is not None:
        gnn_initial_state = {k: v.clone() for k, v in gnn_model.state_dict().items()}
    if adapter_model is not None:
        adapter_initial_state = {k: v.clone() for k, v in adapter_model.state_dict().items()}
    
    if hasattr(model.config, 'n_embd'):
        hidden_dim = model.config.n_embd
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        hidden_dim = 4096
    
    api_node_idx = 0
    if gnn_model is not None and g is not None:
        if 'api' in node_indices:
            api_node_idx = node_indices['api']
        else:
            train_log(f"Warning: API node not found in graph! Using node 0 as fallback.")
            api_node_idx = 0
    
    for window_idx, (win_start, win_end, hit_count) in enumerate(focus_windows):
        train_log(f"\n--- Window {window_idx}: tokens [{win_start}:{win_end}], hit={hit_count} ---")
        
        current_target_ids = target_ids[win_start:win_end]
        current_weights = token_weights[win_start:win_end]
        
        if win_end > 1:
            prefix_answer = target_ids[:win_end - 1]
            input_ids = torch.cat([
                base_input_ids,
                prefix_answer.unsqueeze(0)
            ], dim=1)
        else:
            input_ids = base_input_ids
        
        ex_len = input_ids.shape[1]
        
        rewriting_targets = torch.full((1, ex_len), -100, dtype=torch.long, device="cuda")
        target_start_in_seq = ex_len - len(current_target_ids)
        rewriting_targets[0, target_start_in_seq:ex_len] = current_target_ids
        
        window_weight_mask = torch.zeros(1, ex_len, device="cuda")
        window_weight_mask[0, target_start_in_seq:ex_len] = current_weights
        
        lookup_idxs = [target_start_in_seq]
        loss_layer = max(hparams.v_loss_layer, layer)
        
        delta = torch.zeros((hidden_dim,), requires_grad=True, device="cuda")
        
        target_init = None
        gnn_delta = None
        
        _is_skip = hit_count == 0 and not getattr(hparams, 'ablate_uniform_window', False)
        if _is_skip:
            num_steps = 5
            current_lr = hparams.v_lr * 0.1
            use_gnn_this_window = False
            train_log(f"  [Skip mode] No critical tokens, steps={num_steps}")
        else:
            num_steps = hparams.v_num_grad_steps
            current_lr = hparams.v_lr
            use_gnn_this_window = True
            train_log(f"  [Focus mode] hit={hit_count} (ablate={getattr(hparams, 'ablate_uniform_window', False)}), steps={num_steps}, lr={current_lr}")
        
        
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init, gnn_delta
            
            if cur_layer == hparams.layer_module_tmp.format(layer):
                if isinstance(cur_out, tuple):
                    out_tensor = cur_out[0]
                    is_tuple = True
                else:
                    out_tensor = cur_out
                    is_tuple = False
                
                if out_tensor.dim() == 2:
                    out_tensor = out_tensor.unsqueeze(0)
                
                if target_init is None:
                    target_init = out_tensor[0, lookup_idxs[0]].detach().clone()
                
                total_delta = delta
                
                if gnn_delta is not None:
                    total_delta = total_delta + gnn_delta * hparams.gnn_delta_scale
                
                out_tensor[0, lookup_idxs[0], :] += total_delta
                
                if is_tuple:
                    cur_out = (out_tensor,) + cur_out[1:] if len(cur_out) > 1 else (out_tensor,)
                else:
                    cur_out = out_tensor
            
            return cur_out
        
        params_to_optimize = [{'params': [delta], 'lr': current_lr}]
        gnn_lr = None
        adapter_lr = None
        
        if gnn_model is not None and use_gnn_this_window:
            gnn_model.load_state_dict(gnn_initial_state)
            gnn_model.train()
            
            gnn_lr = getattr(hparams, 'gnn_lr', hparams.v_lr * 0.1)
            params_to_optimize.append({
                'params': gnn_model.parameters(),
                'lr': gnn_lr,
                'weight_decay': getattr(hparams, 'gnn_weight_decay', 1e-1)
            })
                
        if adapter_model is not None and use_gnn_this_window:
            adapter_model.load_state_dict(adapter_initial_state)
            adapter_model.train()
            
            adapter_lr = getattr(hparams, 'adapter_lr', hparams.v_lr * 0.05)
            params_to_optimize.append({
                'params': adapter_model.parameters(),
                'lr': adapter_lr,
                'weight_decay': getattr(hparams, 'adapter_weight_decay', 1e-1)
            })
        
        opt = torch.optim.AdamW(params_to_optimize)
        nethook.set_requires_grad(False, model)
        train_log(
            f"  Train setup: steps={num_steps}, lr={current_lr}, "
            f"gnn_lr={gnn_lr if gnn_lr is not None else 'off'}, "
            f"adapter_lr={adapter_lr if adapter_lr is not None else 'off'}, "
            f"delta_dim={hidden_dim}"
        )
        
        min_loss = None
        min_loss_state = None
        patience = 0
        use_early_stopping = getattr(hparams, 'use_early_stopping', False)
        max_patience = getattr(hparams, 'early_stop_patience', 5) if use_early_stopping else float('inf')
        early_stop_threshold = getattr(hparams, 'early_stop_threshold', 0.01) if use_early_stopping else 0.0
        
        stop_reason = "completed"
        for it in range(num_steps):
            opt.zero_grad()
            
            if gnn_model is not None and g is not None and use_gnn_this_window:
                node_feats = g.ndata['feat']
                gnn_delta = gnn_model(g, node_feats, init_rel_emb, api_node_idx)
                
                if adapter_model is not None:
                    gnn_delta = adapter_model(gnn_delta)
                
                if target_init is not None:
                    max_gnn_norm = hparams.clamp_norm_factor * target_init.norm() * 0.5
                    gnn_norm = gnn_delta.norm()
                    if gnn_norm > max_gnn_norm:
                        scale_factor = max_gnn_norm / (gnn_norm + 1e-8)
                        gnn_delta = gnn_delta * scale_factor
            
            # Forward
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(input_ids).logits
            
            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            if output.dim() == 2:
                output = output.unsqueeze(0)
            if output.shape[0] != input_ids.shape[0]:
                if output.shape[1] == input_ids.shape[0]:
                    output = output.transpose(0, 1)
            if output.shape[1] != rewriting_targets.shape[1]:
                if output.shape[0] == rewriting_targets.shape[1]:
                    output = output.transpose(0, 1)
            
            full_repr = output
            log_probs = torch.log_softmax(
                ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
                dim=2
            )
            
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()
            
            weighted_mask = mask * window_weight_mask
            nll_loss_each = -(loss * weighted_mask.to(loss.device)).sum(1) / weighted_mask.sum(1).clamp(min=1)
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
            
            total_loss = nll_loss + weight_decay.to(nll_loss.device)
            
            if min_loss is None or total_loss < min_loss - 0.005:
                min_loss = total_loss.item()
                min_loss_state = {
                    'delta': delta.detach().clone(),
                    'gnn_delta': gnn_delta.detach().clone() if gnn_delta is not None else None,
                }
                patience = 0
            else:
                if it % 3 == 0:
                    patience += 1
            
            if use_early_stopping and total_loss < early_stop_threshold:
                train_log(f"  Early stop: loss < {early_stop_threshold}")
                stop_reason = "loss_threshold"
                break
            
            if use_early_stopping and patience >= max_patience and it >= 5:
                train_log(f"  Early stop: no improvement for {max_patience} steps")
                stop_reason = "no_improve"
                break
            
            if it == num_steps - 1:
                break
            
            total_loss.backward()
            
            if it % 5 == 0 or it == num_steps - 1:
                gnn_norm = gnn_delta.norm().item() if gnn_delta is not None else 0
                gnn_grad_norm = 0.0
                if gnn_model is not None and use_gnn_this_window:
                    gnn_params = list(gnn_model.parameters())
                    if len(gnn_params) > 0 and gnn_params[0].grad is not None:
                        gnn_grad_norm = sum(p.grad.norm().item() for p in gnn_params if p.grad is not None) / len(gnn_params)
                
                delta_grad_norm = delta.grad.norm().item() if delta.grad is not None else 0
                is_clamped = delta.norm().item() >= (hparams.clamp_norm_factor * target_init.norm().item() * 0.999)
                clamp_str = " [CLAMPED]" if is_clamped else ""
                train_log(
                    f"  Step {it}: loss={total_loss.item():.4f} "
                    f"(nll={nll_loss.item():.4f}, wd={weight_decay.item():.4f}) "
                    f"delta={delta.norm().item():.4f} (grad={delta_grad_norm:.4f}){clamp_str}, "
                    f"gnn={gnn_norm:.4f} (grad={gnn_grad_norm:.4f}), "
                    f"prob={torch.exp(-nll_loss_each).mean().item():.4f}"
                )
            
            torch.nn.utils.clip_grad_norm_([delta], max_norm=5.0)
            if gnn_model is not None and use_gnn_this_window:
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=3.0)
            if adapter_model is not None and use_gnn_this_window:
                torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), max_norm=3.0)
            
            opt.step()
            
            if target_init is not None:
                with torch.no_grad():
                    max_norm = hparams.clamp_norm_factor * target_init.norm()
                    if delta.norm() > max_norm:
                        delta.data = delta.data * (max_norm / delta.norm())
            
            if it % 10 == 0:
                torch.cuda.empty_cache()
        
        if min_loss_state is not None and total_loss.item() > min_loss + 0.1:
            delta = min_loss_state['delta']
            if min_loss_state['gnn_delta'] is not None:
                gnn_delta = min_loss_state['gnn_delta']
        
        if min_loss is not None:
            train_log(f"  Window done: min_loss={min_loss:.4f}, reason={stop_reason}")
        
        final_delta = delta.detach()
        if gnn_delta is not None:
            final_delta = final_delta + gnn_delta.detach() * hparams.gnn_delta_scale
        
        target = target_init + final_delta
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])
        
        gnn_norm = gnn_delta.norm().item() if gnn_delta is not None else 0.0
        log(
            f"  Window {window_idx} done: init={target_init.norm():.4f}, "
            f"delta={delta.norm().item():.4f}, gnn={gnn_norm:.4f}"
        )
        
        gnn_delta = None
        del delta, final_delta
        torch.cuda.empty_cache()
    
    if gnn_model is not None:
        gnn_model.eval()
    
    if gnn_model is not None:
        del gnn_model
    if adapter_model is not None:
        del adapter_model
    if g is not None:
        del g
    torch.cuda.empty_cache()
    
    return all_idxs, all_target

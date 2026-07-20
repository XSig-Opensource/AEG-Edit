"""
AlphaEdit_ARE_AEG_main.py

AEG-Edit target-construction wrapper on top of AlphaEdit-ARE.
This file keeps the existing editing logic and standardizes the public naming.
"""

import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import nethook
from util.globals import *
import torch.optim as optim
from util.layer_stats import layer_stats
import argparse

import numpy as np
import os

from .AlphaEdit_ARE_AEG_hparams import AlphaEditAREAEGHyperParams
from .compute_aeg_edit_targets import compute_aeg_edit_targets

COV_CACHE = {}


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: AlphaEditAREAEGHyperParams,
    layer: int,
    idxs_dict: dict,
):
    """ K vector(vector)"""
    input_ids = tok(batch_data, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = model(**input_ids)
            zs_out = tr.output
    
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list = []
    for k, idxs in idxs_dict.items():
        for idx in idxs:
            zs_out_list.append(zs_out[k, idx])
    zs_out = torch.stack(zs_out_list, dim=1)
    return zs_out


def apply_alphaedit_are_aeg_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: AlphaEditAREAEGHyperParams,
    batch_data: list,
    P=None,
    ex_data: list = None,
    graph_data_list: list = None,
    dataset_type: str = "rustevo",
):
    """
    AlphaEdit-ARE + AEG-Edit target construction model
    
    Supportdata:
    - rustevo: Rust API evolution
    - pyevo: Python API evolution

    
    Args:
        model: editmodel
        tok: tokenizer
        hparams: hyperparameters(with AEG config)
        batch_data: editdata
        P: matrix( AlphaEdit)
        graph_data_list: Buildgraphdata( batch_data corresponding)
        dataset_type: datatype, tokenextractstrategy
        
    Returns:
        weights_copy: originalweights()
    """
    
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    z_layer = hparams.layers[-1]
    all_zs_list = []
    idxs_dict = {}
    
    for k, data in enumerate(batch_data):
        print(f"\nProcessing sample {k+1}/{len(batch_data)}")
        
        graph_data = None
        if graph_data_list is not None and k < len(graph_data_list):
            graph_data = graph_data_list[k]
        
        idxs_list, zs_list = compute_aeg_edit_targets(
            model,
            tok,
            data,
            z_layer,
            hparams,
            graph_data=graph_data,
            dataset_type=dataset_type,
        )
        
        all_zs_list.extend(zs_list)
        idxs_dict[k] = idxs_list
    
    zs = torch.stack(all_zs_list, dim=1)
    
    batch_question_ans = [
        i['question'] + i['answer'] for i in batch_data
    ]
    
    for i, layer in enumerate(hparams.layers):
        print(f"\nUpdating layer {layer}")
        
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.rewrite_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input
                layer_out_ks = tr.output
        
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        
        cur_zs = compute_ks(model, tok, batch_question_ans, hparams, z_layer, idxs_dict)
        
        targets = zs - cur_zs
        print(f"z error: {torch.linalg.norm(targets, dim=0).mean():.4f}")
        
        ks_list = []
        kp_list = []
        for k, idxs in idxs_dict.items():
            all_idxs = set(range(len(layer_in_ks[k])))
            unselected_idxs = list(all_idxs - set(idxs))
            for idx in idxs:
                ks_list.append(layer_in_ks[k, idx])
            for unselected_idx in unselected_idxs:
                kp_list.append(layer_in_ks[k, unselected_idx])
        
        layer_ks = torch.stack(ks_list, dim=1)
        layer_kp = torch.stack(kp_list, dim=1)
        
        resid = targets / (len(hparams.layers) - i)
        
        upd_matrix = torch.linalg.solve(
            P[i, :, :].cuda() @ (layer_ks @ layer_ks.T + layer_kp @ layer_kp.T) + 
            hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda"),
            P[i, :, :].cuda() @ layer_ks @ resid.T
        )
        
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        print(f"orig norm: {torch.linalg.norm(weights[weight_name]):.4f}")
        print(f"upd norm: {torch.linalg.norm(upd_matrix):.4f}")
        
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
        
        for x in [layer_ks, layer_kp, cur_zs, targets, layer_in_ks, layer_out_ks]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    print("\nAlphaEdit_ARE_AEG completed")
    return weights_copy


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Get covariance statistics.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    matrixmatchweights
    """
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed does not match original weight shape. "
            "Check for bugs in the code?"
        )


class AlphaEditAREAGEEditor:
    """
    AlphaEdit + ARE + AEG edit
    
    Model editing
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        hparams: AlphaEditAREAEGHyperParams,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams
        
        self.P = self._compute_projection_matrices()
        
        self.aeg_enhancer = None
        
    def _compute_projection_matrices(self):
        """
        matrix P
        
        Core:Usematrix SVD ,
        edit, model.
        
        AlphaEdit , Usematrix!
        """
        hidden_size = self.model.config.hidden_size
        P = torch.zeros(
            (len(self.hparams.layers), hidden_size, hidden_size), 
            dtype=torch.float
        )
        
        print("Computing null-space projection matrices...")
        
        for i, layer in enumerate(self.hparams.layers):
            print(f"  Layer {layer}: computing covariance and SVD...")
            
            try:
                cov = get_cov(
                    self.model,
                    self.tokenizer,
                    self.hparams.rewrite_module_tmp.format(layer),
                    self.hparams.mom2_dataset,
                    self.hparams.mom2_n_samples,
                    self.hparams.mom2_dtype,
                    force_recompute=False,
                ).cpu()
                
                U, S, _ = torch.linalg.svd(cov, full_matrices=False)
                
                threshold = self.hparams.nullspace_threshold
                small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
                
                if len(small_singular_indices) > 0:
                    U_null = U[:, small_singular_indices]
                    P[i] = U_null @ U_null.T
                    print(f"    Found {len(small_singular_indices)} null-space directions")
                else:
                    print(f"    Warning: No null-space found, using identity")
                    P[i] = torch.eye(hidden_size, dtype=torch.float)
                    
            except Exception as e:
                print(f"    Error computing P for layer {layer}: {e}")
                print(f"    Falling back to identity matrix")
                P[i] = torch.eye(hidden_size, dtype=torch.float)
        
        print(f"Projection matrices computed for {len(self.hparams.layers)} layers")
        return P
    
    def edit(
        self,
        samples: list,
        graph_inputs: list = None,
        gnn_model = None,
    ) -> dict:
        """
        Model editing
        
        Args:
            samples: editsample
            graph_inputs: Buildgraph [(g, node_indices, rel_emb), ...]
            gnn_model: GNN model(graph)
            
        Returns:
            weights_copy: originalweights
        """
        if self.hparams.use_gnn and graph_inputs is not None:
            if self.aeg_enhancer is None:
                try:
                    self.aeg_enhancer = AEGEnhancer(
                        self.hparams, 
                        self.model,
                        gnn_model=gnn_model
                    )
                except Exception as e:
                    print(f"Warning: Failed to init AEG enhancer: {e}")
                    self.aeg_enhancer = None
            
            if self.aeg_enhancer is not None:
                self.aeg_enhancer.set_graph_inputs(graph_inputs)
        
        weights_copy = apply_alphaedit_are_aeg_to_model(
            model=self.model,
            tok=self.tokenizer,
            hparams=self.hparams,
            batch_data=samples,
            P=self.P,
        )
        
        return weights_copy
    
    def restore(self, weights_copy: dict):
        """Restore original weights."""
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to("cuda")

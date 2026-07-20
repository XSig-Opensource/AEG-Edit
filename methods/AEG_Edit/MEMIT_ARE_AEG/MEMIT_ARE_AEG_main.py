"""
MEMIT_ARE_AEG_main.py

AEG-Edit target-construction wrapper on top of MEMIT-ARE.
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

from .MEMIT_ARE_AEG_hparams import MEMITAREAEGHyperParams
from .compute_aeg_edit_targets import compute_aeg_edit_targets

COV_CACHE = {}


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: MEMITAREAEGHyperParams,
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


def apply_memit_are_aeg_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: MEMITAREAEGHyperParams,
    batch_data: list,
    P=None,
    ex_data: list = None,
    graph_data_list: list = None,
    dataset_type: str = "rustevo",
):
    """
    Apply MEMIT-ARE + AEG-Edit target construction constraints to the model.
    
    Args:
        model: The model to be edited.
        tok: Tokenizer.
        hparams: Hyperparameters including AEG config.
        batch_data: Batch of data for editing.
        P: Null-space projection matrix (used for AlphaEdit variant).
        graph_data_list: Prebuilt graph structures corresponding to batch_data.
        dataset_type: Selected dataset type used for critical token extraction.
        
    Returns:
        weights_copy: Original weights backup to be used for restoration.
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
        print(f"z error: {torch.linalg.norm(targets, dim=1).mean():.4f}")
        
        force_recompute = False
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )
        
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
        
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov + layer_kp @ layer_kp.T + layer_ks @ layer_ks.T,
            layer_ks
        )
        
        resid = targets / (len(hparams.layers) - i)
        upd_matrix = resid @ adj_k.T
        
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        print(f"orig norm: {torch.linalg.norm(weights[weight_name]):.4f}")
        print(f"upd norm: {torch.linalg.norm(upd_matrix):.4f}")
        
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
        
        cov.cpu()
        for x in [layer_ks, layer_kp, cur_zs, targets, layer_in_ks, layer_out_ks]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    print("\nMEMIT_ARE_AEG completed")
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
    Match update matrix shape to original weights.
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


class MEMITAREAGEEditor:
    """
    MEMIT + AEG wrapper class for simple model editing interface.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        hparams: MEMITAREAEGHyperParams,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams
        
        self.aeg_enhancer = None
        
    def edit(
        self,
        samples: list,
        ex_data: list = None,
        graph_data_list: list = None,
        graph_inputs: list = None,
        gnn_model = None,
    ) -> dict:
        """
        Execute model editing.
        
        Args:
            samples: List of edit samples.
            ex_data: Extra data for stability injection (unused by MEMIT).
            graph_data_list: Prebuilt graph data list.
            graph_inputs: Deprecated inline graph building list.
            gnn_model: External GNN model (deprecated).
            
        Returns:
            weights_copy: Backup of original weights.
        """
        # if self.hparams.use_gnn and graph_inputs is not None:
        #     if self.aeg_enhancer is None:
        #         try:
        #             self.aeg_enhancer = AEGEnhancer(
        #                 self.hparams, 
        #                 self.model,
        #                 gnn_model=gnn_model
        #             )
        #         except Exception as e:
        #             print(f"Warning: Failed to init AEG enhancer: {e}")
        #             self.aeg_enhancer = None
        #     
        #     if self.aeg_enhancer is not None:
        #         self.aeg_enhancer.set_graph_inputs(graph_inputs)
        
        weights_copy = apply_memit_are_aeg_to_model(
            model=self.model,
            tok=self.tokenizer,
            hparams=self.hparams,
            batch_data=samples,
            ex_data=ex_data,
            P=None,
            graph_data_list=graph_data_list,
        )
        
        return weights_copy
    
    def restore(self, weights_copy: dict):
        """Restore original weights."""
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to("cuda")

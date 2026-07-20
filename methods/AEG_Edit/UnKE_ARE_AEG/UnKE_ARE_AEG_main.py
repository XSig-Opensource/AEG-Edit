"""
unke_ARE_AEG_main.py

AEG-Edit target-construction wrapper on top of UnKE-ARE.
This file keeps the existing editing logic and standardizes the public naming.
"""

import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import nethook
from util.globals import *
import torch.optim as optim
import argparse

import numpy as np
import os

from .UnKE_ARE_AEG_hparams import unkeAREAEGHyperParams
from .compute_aeg_edit_targets import compute_aeg_edit_targets

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
    """getparameter( layernorm weight decay)"""
    param_optimizer = list(model.named_parameters())
    no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    optimizer_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0
        },
    ]
    return optimizer_parameters


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeAREAEGHyperParams,
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
    zs_out_dict = {}
    for k, idxs in idxs_dict.items():
        zs_out_list = []
        for idx in idxs:
            zs_out_list.append(zs_out[k, idx])
        zs_out_dict[k] = zs_out_list
    return zs_out_dict


def apply_unke_are_aeg_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: unkeAREAEGHyperParams,
    batch_data: list,
    P=None,
    ex_data: list = None,
    graph_data_list: list = None,
    dataset_type: str = "rustevo",
):
    """
    UnKE-ARE + AEG-Edit target construction model
    
    Args:
        model: editmodel
        tok: tokenizer
        hparams: hyperparameters(with AEG config)
        batch_data: editdata
        P: matrix(Use, only)
        ex_data: stabledata
        graph_data_list: Buildgraphdata( batch_data corresponding)
        dataset_type: datatype, selecttokenextractstrategy
        
    Returns:
        weights_copy: originalweights()
    """
    
    preserve_params = []
    for name, params in model.named_parameters():
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    
    weights = {
        param: nethook.get_parameter(model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    z_layer = hparams.layers[-1]
    zs_dict = {}
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
        
        idxs_dict[k] = idxs_list
        zs_dict[k] = zs_list
    
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
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input
                layer_out_ks = tr.output
        
        layer_in_ks = layer_in_ks[0] if type(layer_in_ks) is tuple else layer_in_ks
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        
        cur_zs_dict = compute_ks(model, tok, batch_question_ans, hparams, z_layer, idxs_dict)
        
        targets_dict = {}
        for k, cur_zs_list in cur_zs_dict.items():
            zs_list = zs_dict[k]
            targets_list = [(a - b) / (len(hparams.layers) - i) for a, b in zip(zs_list, cur_zs_list)]
            targets_dict[k] = targets_list
        
        all_targets = []
        for k in targets_dict:
            all_targets.extend(targets_dict[k])
        if all_targets:
            mean_error = sum(torch.linalg.norm(t).item() for t in all_targets) / len(all_targets)
            print(f"z error: {mean_error:.4f}")
        
        if ex_data is None:
            ex_data = []
        
        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        
        stat_in = stat_in[0] if type(stat_in) is tuple else stat_in
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out
        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n, m in _layer.named_parameters():
            m.requires_grad = True
        
        params = get_optimizer_params(_layer, hparams.lr)
        optimizer = optim.AdamW(params, lr=hparams.lr, eps=1e-8, betas=(0.9, 0.999))
        
        for k, idxs_list in idxs_dict.items():
            for j, idx in enumerate(idxs_list):
                resid = targets_dict[k][j]
                layer_out_ks[k, idx] += resid
        
        if hparams.model_name in ['Llama3-8B-Instruct', 'Llama3.1-8B-Instruct']:
            input_causal_mask, input_position_ids, input_cache_position, input_position_embeddings = get_causal_mask(model, layer_in_ks, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids, ex_cache_position, ex_position_embeddings = get_causal_mask(model, stat_in, ex_tok['attention_mask'])
        elif hparams.model_name == 'Qwen2.5-7B-Instruct':
            input_causal_mask, input_position_ids, input_position_embeddings = get_qwen2_causal_mask(model, layer_in_ks, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids, ex_position_embeddings = get_qwen2_causal_mask(model, stat_in, ex_tok['attention_mask'])
        
        for step in range(hparams.optim_num_step):
            optimizer.zero_grad()
            
            if hparams.model_name == 'Qwen2.5-7B-Instruct':
                if stat_in.shape[0] > 1:
                    ex_outputs = []
                    for i_batch in range(stat_in.shape[0]):
                        single_output = _layer(
                            stat_in[i_batch:i_batch+1],
                            attention_mask=ex_causal_mask[i_batch:i_batch+1] if ex_causal_mask is not None else None,
                            position_ids=ex_position_ids[i_batch:i_batch+1],
                            position_embeddings=(ex_position_embeddings[0][i_batch:i_batch+1], ex_position_embeddings[1][i_batch:i_batch+1])
                        )[0]
                        if single_output.dim() == 2:
                            single_output = single_output.unsqueeze(0)
                        ex_outputs.append(single_output)
                    ex_output = torch.cat(ex_outputs, dim=0)
                else:
                    ex_output = _layer(stat_in, attention_mask=ex_causal_mask, position_ids=ex_position_ids, position_embeddings=ex_position_embeddings)[0]
                    if ex_output.dim() == 2:
                        ex_output = ex_output.unsqueeze(0)
                
                input_output = _layer(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids, position_embeddings=input_position_embeddings)[0]
                if input_output.dim() == 2:
                    input_output = input_output.unsqueeze(0)
                
                loss = criterion(ex_output, stat_out) + criterion(input_output, layer_out_ks)
            
            elif hparams.model_name in ['Llama3-8B-Instruct', 'Llama3.1-8B-Instruct']:
                if stat_in.shape[0] > 1:
                    ex_outputs = []
                    for i_batch in range(stat_in.shape[0]):
                        single_output = _layer(
                            stat_in[i_batch:i_batch+1],
                            attention_mask=ex_causal_mask[i_batch:i_batch+1] if ex_causal_mask is not None else None,
                            position_ids=ex_position_ids[i_batch:i_batch+1],
                            cache_position=ex_cache_position,
                            position_embeddings=(ex_position_embeddings[0][i_batch:i_batch+1], ex_position_embeddings[1][i_batch:i_batch+1])
                        )[0]
                        if single_output.dim() == 2:
                            single_output = single_output.unsqueeze(0)
                        ex_outputs.append(single_output)
                    ex_output = torch.cat(ex_outputs, dim=0)
                else:
                    ex_output = _layer(stat_in, attention_mask=ex_causal_mask, position_ids=ex_position_ids, cache_position=ex_cache_position, position_embeddings=ex_position_embeddings)[0]
                    if ex_output.dim() == 2:
                        ex_output = ex_output.unsqueeze(0)
                
                input_output = _layer(layer_in_ks, attention_mask=input_causal_mask, position_ids=input_position_ids, cache_position=input_cache_position, position_embeddings=input_position_embeddings)[0]
                if input_output.dim() == 2:
                    input_output = input_output.unsqueeze(0)
                
                loss = criterion(ex_output, stat_out) + criterion(input_output, layer_out_ks)
            
            loss.backward(retain_graph=True)
            optimizer.step()
        
        for x in [layer_in_ks, layer_out_ks, stat_in, stat_out]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    print("\nUnKE_ARE_AEG completed")
    return weights_copy


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


class unkeAREAGEEditor:
    """
    unke + ARE + AEG edit
    
    Model editing
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        hparams: unkeAREAEGHyperParams,
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
        Model editing
        
        Args:
            samples: editsample
            ex_data: stabledata
            graph_data_list: Buildgraphdata
            graph_inputs: Buildgraph [(g, node_indices, rel_emb), ...](deprecated)
            gnn_model: GNN model(graph)(deprecated)
            
        Returns:
            weights_copy: originalweights
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
        
        weights_copy = apply_unke_are_aeg_to_model(
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


def get_qwen2_causal_mask(model, input_tensor, attention_mask, past_key_values_length=0):
    """get Qwen2 causal mask"""
    device = input_tensor.device
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask,
        (input_tensor.shape[0], input_tensor.shape[1]),
        input_tensor,
        0,
    )
    
    position_embeddings = model.model.rotary_emb(input_tensor, position_ids)

    return attention_mask, position_ids, position_embeddings


def get_causal_mask(model, input_tensor, attention_mask):
    """get Llama causal mask"""
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    batch_size = input_tensor.shape[0]
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    
    position_embeddings = model.model.rotary_emb(input_tensor, position_ids)
    
    return causal_mask, position_ids, cache_position, position_embeddings

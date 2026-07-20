"""
GRACE module - AEG-Edit
"""
from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE
from .grace_hparams import GraceHyperParams
from .utils import tokenize, parent_module, brackets_to_periods
from util import nethook

_grace_editors = {}


def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        hparams: GraceHyperParams,
        batch_data: list,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Any]:
    """
    GRACE editmodel
    
    Args:
        model: editmodel
        tok: tokenizer
        hparams: GRACE hyperparameters
        batch_data: editdata, with 'question' 'answer'
        copy: model
        return_orig_weights: returnoriginalweights
        keep_original_weight: originalweights
    
    Returns:
        weights_copy: withinfo
    """
    global _grace_editors
    
    request = batch_data[0]
    
    if 'prompt' not in request:
        request['prompt'] = request.get('question', '')
    if 'target_new' not in request:
        request['target_new'] = request.get('answer', '')
    
    if copy:
        model = deepcopy(model)
    
    device = torch.device(f'cuda:{hparams.device}')
    
    model_id = id(model)
    if model_id in _grace_editors:
        _grace_editors[model_id].reset_layer()
        del _grace_editors[model_id]
    
    editor = GRACE(model=model, config=hparams, device=device)
    
    tokens = tokenize(request, tokenizer=tok, device=device)
    
    if 'token_type_ids' in tokens:
        tokens.pop('token_type_ids')
    
    edit_id = request.get('target_new', str(id(request)))
    editor.edit(config=hparams, tokens=tokens, edit_id=edit_id)

    adapter = editor._get_layer()
    adapter.key_id = -1
    adapter.ensure_replace_token_loc = False

    _grace_editors[model_id] = editor
    
    weights_copy = {
        '_grace_model_id': model_id,
        '_grace_layer': editor.layer,
    }
    
    if editor.original_layer is not None:
        layer_name = editor.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(model, brackets_to_periods(editor.layer))
        for name, param in editor.original_layer.named_parameters():
            full_name = f"{editor.layer}.{name}"
            weights_copy[full_name] = param.detach().clone()
            
    if keep_original_weight:
        editor.reset_layer()
        if model_id in _grace_editors:
            del _grace_editors[model_id]

    return weights_copy


def restore_grace_model(model: AutoModelForCausalLM, weights_copy: Dict[str, Any]):
    """
    GRACE editmodel
    
    Args:
        model: model
        weights_copy: apply_grace_to_model returninfo
    """
    global _grace_editors
    
    model_id = weights_copy.get('_grace_model_id')
    if model_id is not None and model_id in _grace_editors:
        _grace_editors[model_id].reset_layer()
        del _grace_editors[model_id]


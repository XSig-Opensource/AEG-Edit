"""
rome_main.py

ROME(Rank-One Model Editing)implement
Paper: "Locating and Editing Factual Associations in GPT" (Meng et al., 2022)

MEMIT Core:
- ROME: , rank-1 , sample
- MEMIT: , ,

Formula(sample):
  W_new = W + (z* - z_cur) ⊗ adj_k
  adj_k = C^{-1}k / (k^T C^{-1}k)

:
  z* = Target layersOutputvector ( compute_z )
  z_cur = token Output
  k = rewrite_module token Input(vector)
  C = k (matrix)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from methods.Baselines.MEMIT_ARE.compute_z import compute_z
from methods.Baselines.MEMIT_ARE.memit_ARE_main import get_cov, upd_matrix_match_shape
from util import nethook
from util.globals import *

from .rome_hparams import ROMEHyperParams

COV_CACHE = {}


def compute_ks_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    questions: list,
    hparams: ROMEHyperParams,
    layer: int,
):
    """
    get layer_module_tmp.format(layer) sample token Output( z_cur).
    return (zs_out, idxs), memit_main.compute_ks .
    """
    input_ids = tok(questions, padding=True, return_tensors="pt").to("cuda")
    idxs = [int(mask.sum()) - 1 for mask in input_ids["attention_mask"]]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=False,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = model(**input_ids)
            zs_out = tr.output

    zs_out = zs_out[0] if isinstance(zs_out, tuple) else zs_out
    zs_list = [zs_out[i, idxs[i]] for i in range(len(idxs))]
    return torch.stack(zs_list, dim=0), idxs


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: ROMEHyperParams,
    batch_data: list,
) -> dict:
    """
    batch_data sample ROME rank-1 .

    Returns:
        weights_copy: Saveweights, model
    """
    layer = hparams.layer
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"

    weights = {
        weight_name: nethook.get_parameter(model, weight_name)
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
        force_recompute=False,
    )

    for data in batch_data:
        z_star = compute_z(model, tok, data, layer, hparams)  # (d_model,)

        question = data["question"]
        context_tok = tok([question], return_tensors="pt").to("cuda")
        last_idx = int(context_tok["attention_mask"].sum()) - 1

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.rewrite_module_tmp.format(layer),
                retain_input=True,
                retain_output=False,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**context_tok)
                layer_in = tr.input

        if isinstance(layer_in, tuple):
            layer_in = layer_in[0]
        k = layer_in[0, last_idx]  # (d_ffn,)

        cur_zs, _ = compute_ks_rome(model, tok, [question], hparams, layer)
        z_cur = cur_zs[0]  # (d_model,)

        target = z_star - z_cur
        print(f"  z error (ROME): {target.norm().item():.4f}")

        C_inv_k = torch.linalg.solve(cov, k.unsqueeze(1))  # (d_ffn, 1)
        denom = (k @ C_inv_k.squeeze()).item()
        adj_k = C_inv_k.squeeze() / denom  # (d_ffn,)

        upd_matrix = target.unsqueeze(1) @ adj_k.unsqueeze(0)
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print(f"  orig norm: {weights[weight_name].norm().item():.4f}")
        print(f"  upd  norm: {upd_matrix.norm().item():.6f}")

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()

        # Free GPU memory
        for x in [C_inv_k, adj_k, upd_matrix, k, target, z_star, z_cur, cur_zs, layer_in]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    cov.cpu()
    del cov
    torch.cuda.empty_cache()

    return weights_copy

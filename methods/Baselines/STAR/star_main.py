from __future__ import annotations

import torch
from torch import nn

from .star_hparams import STARHyperParams
from .utils import Seme, freeze_params, get_attr


def _detect_attrs(model_name: str) -> dict:
    lowered = model_name.lower()
    if "llama" in lowered or "qwen" in lowered:
        return {
            "layers": "model.layers",
            "ff_act": "mlp.act_fn",
            "ff_input": "mlp.up_proj",
            "ff_output": "mlp.down_proj",
            "embedding": "model.embed_tokens",
            "lm_head": "lm_head",
        }
    raise ValueError(f"STAR does not support model architecture: {model_name}")


def _compute_sss(
    lm_head_pinv: torch.Tensor,
    target_label: int,
    argmax_label: int,
) -> torch.Tensor:
    target_embed = lm_head_pinv[target_label].detach()
    argmax_embed = lm_head_pinv[argmax_label].detach()
    semantic_steer = target_embed - argmax_embed
    iqr = torch.quantile(semantic_steer, 0.75) - torch.quantile(semantic_steer, 0.25)
    semantic_steer = semantic_steer / (iqr + 1e-8)

    if semantic_steer.dim() == 1:
        semantic_steer = semantic_steer.unsqueeze(0)
    return semantic_steer


def _obtain_layer_acts(model, attrs: dict, input_ids: torch.Tensor,
                       label_count: int):
    layer_outputs = {}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            layer_outputs[idx] = output[:, -label_count:, :].flatten(0, 1)

        return hook

    handles = []
    layers = get_attr(model, attrs["layers"])
    for idx, layer in enumerate(layers):
        activation = get_attr(layer, attrs["ff_act"])
        handles.append(activation.register_forward_hook(make_hook(idx)))

    try:
        out = model(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
    finally:
        for handle in handles:
            handle.remove()

    return layer_outputs, logits


def _do_repair(
    model,
    attrs: dict,
    input_ids: torch.Tensor,
    target_label_id: int,
    scaled_steer: torch.Tensor,
    trainable_layers: list[int] | None,
    optimizer: Seme,
):
    layer_outputs, logits = _obtain_layer_acts(model, attrs, input_ids, 1)

    layer_deltas = {}
    for layer_idx, activations in layer_outputs.items():
        z = activations.to(torch.float)
        steer = scaled_steer.to(torch.float)
        solution = torch.linalg.lstsq(z, steer).solution
        layer_deltas[layer_idx] = solution.T

    optimizer.update_delta(layer_deltas, trainable_layers)
    optimizer.zero_grad()

    target = torch.tensor([target_label_id], device=logits.device)
    last_logits = logits[:, -1, :]
    loss = nn.functional.cross_entropy(last_logits, target)
    loss.backward()
    optimizer.step()


def _select_trainable_layers(n_layers: int, mode: str) -> list[int] | None:
    if mode == "all":
        return None

    amount_map = {
        "one": 1,
        "1quarter": max(1, n_layers // 4),
        "2quarter": max(1, n_layers * 2 // 4),
        "3quarter": max(1, n_layers * 3 // 4),
    }
    n_train = amount_map.get(mode, max(1, n_layers // 2))
    start = max(0, (n_layers - n_train) // 2)
    return list(range(start, start + n_train))


def apply_STAR_to_model(
    model,
    tok,
    hparams: STARHyperParams,
    batch_data: list,
    P=None,
    ex_data=None,
    graph_data_list=None,
    dataset_type: str = "rustevo",
):
    """Apply STAR semantic target repair to one AnyEdit-format sample."""
    del P, ex_data, graph_data_list, dataset_type

    attrs = _detect_attrs(hparams.model_name)
    device = next(model.parameters()).device

    weights = {
        name: param
        for name, param in model.named_parameters()
        if name.endswith(attrs["ff_output"] + ".weight")
    }
    weights_copy = {name: param.detach().clone() for name, param in weights.items()}

    lm_head_weight = get_attr(model, attrs["lm_head"]).weight.detach().to(torch.float)
    lm_head_pinv = torch.linalg.pinv(lm_head_weight.T)

    data = batch_data[0]
    source = data["question"]
    target = data["answer"]

    target_label_ids = tok(target, add_special_tokens=False)["input_ids"]
    target_label_ids = target_label_ids[: hparams.max_focus_tokens]
    if not target_label_ids:
        return weights_copy

    layers = get_attr(model, attrs["layers"])
    trainable_layers = _select_trainable_layers(
        len(layers),
        hparams.trainable_layers_mode,
    )

    param_groups = freeze_params(model, attrs, trainable_layers)
    optimizer = Seme(param_groups, lr=1e-2 * hparams.scale)

    model.train()
    oracle_prefix_tokens: list[str] = []
    for target_label in target_label_ids:
        target_token = tok.decode([target_label], skip_special_tokens=True)
        prompt = source + "".join(oracle_prefix_tokens)
        enc = tok(
            [prompt],
            add_special_tokens=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            argmax_label = int(model(input_ids=input_ids).logits[0, -1].argmax().item())

        if argmax_label == int(target_label):
            oracle_prefix_tokens.append(target_token)
            continue

        scaled_steer = _compute_sss(
            lm_head_pinv,
            int(target_label),
            int(argmax_label),
        ).to(device)

        for _ in range(hparams.epoch_num):
            _do_repair(
                model,
                attrs,
                input_ids,
                int(target_label),
                scaled_steer,
                trainable_layers,
                optimizer,
            )

        oracle_prefix_tokens.append(target_token)

    model.eval()
    return weights_copy

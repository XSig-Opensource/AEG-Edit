import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def get_attr(module: nn.Module, attrs: str):
    for attr in attrs.split("."):
        module = getattr(module, attr)
    return module


def freeze_params(model: nn.Module, attrs: dict, layers: list[int] | None = None):
    param_groups = []
    layer_prefix = attrs["layers"] + "."

    for name, param in model.named_parameters():
        is_ff_output_weight = name.endswith(attrs["ff_output"] + ".weight")
        should_train = False

        if is_ff_output_weight:
            if layers is None:
                should_train = True
            elif name.startswith(layer_prefix):
                layer_name = name.removeprefix(layer_prefix).split(".", 1)[0]
                should_train = int(layer_name) in layers

        param.requires_grad = should_train
        if should_train:
            param_groups.append({"params": [param]})

    if not param_groups:
        raise ValueError("STAR could not find trainable FFN output weights")

    return param_groups


class Seme(Optimizer):
    """STAR semantic update optimizer."""

    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.defaults["matrix_deltas"] = None
        self.defaults["trainable_layers"] = None

    def update_delta(self, matrix_deltas: dict[int, torch.Tensor],
                     trainable_layers: list[int] | None = None):
        self.defaults["matrix_deltas"] = matrix_deltas
        if trainable_layers is None:
            self.defaults["trainable_layers"] = sorted(matrix_deltas.keys())
        else:
            self.defaults["trainable_layers"] = list(trainable_layers)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        matrix_deltas = self.defaults["matrix_deltas"]
        trainable_layers = self.defaults["trainable_layers"]
        if matrix_deltas is None or trainable_layers is None:
            raise RuntimeError("Seme.update_delta must be called before step")

        trainable_layer_idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if trainable_layer_idx >= len(trainable_layers):
                    raise RuntimeError("More trainable STAR parameters than layer deltas")

                layer_idx = trainable_layers[trainable_layer_idx]
                matrix_delta = matrix_deltas[layer_idx].to(
                    device=param.device,
                    dtype=param.dtype,
                )
                if matrix_delta.shape != param.shape:
                    raise ValueError(
                        f"STAR delta shape {tuple(matrix_delta.shape)} does not "
                        f"match parameter shape {tuple(param.shape)}"
                    )

                grad_sign = param.grad.sign()
                aligned_delta = torch.where(
                    matrix_delta.sign() != -grad_sign,
                    -matrix_delta,
                    matrix_delta,
                )
                param.add_(aligned_delta, alpha=group["lr"])
                trainable_layer_idx += 1

        return loss

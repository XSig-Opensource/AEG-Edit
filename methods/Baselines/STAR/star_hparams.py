from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class STARHyperParams(HyperParams):
    model_name: str
    layers: List[int]
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str = ""
    ln_f_module: str = ""
    lm_head_module: str = ""

    epoch_num: int = 8
    max_focus_tokens: int = 30
    trainable_layers_mode: str = "2quarter"
    scale: int = 1

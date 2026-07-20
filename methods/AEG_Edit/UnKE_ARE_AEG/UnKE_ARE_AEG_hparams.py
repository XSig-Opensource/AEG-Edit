from dataclasses import dataclass
from typing import List, Literal, Optional

from util.hparams import HyperParams


@dataclass
class unkeAREAEGHyperParams(HyperParams):
    """
    UNKE + ARE + AEG hyperparameters
    
    UNKE_ARE AEG graphparameter
    """
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    lr: float
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    optim_num_step: int
    ex_data_num: int
    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    window_size: int
    overlap: int
    
    use_gnn: bool = True
    
    gnn_hidden_dim: int = 4096
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1
    
    gnn_num_grad_steps: int = 25
    gnn_lr: float = 1e-4
    gnn_weight_decay: float = 0.01
    
    gnn_delta_scale: float = 0.1

    use_adapter: bool = False
    adapter_hidden_dim: int = 1024
    adapter_num_layers: int = 2
    adapter_activation: str = "gelu"
    adapter_dropout: float = 0.1
    adapter_lr: float = 1e-1
    adapter_weight_decay: float = 1e-1

    critical_token_weight: float = 3.0
    
    use_early_stopping: bool = False
    early_stop_patience: int = 5
    early_stop_threshold: float = 0.01

    ablate_uniform_window: bool = False

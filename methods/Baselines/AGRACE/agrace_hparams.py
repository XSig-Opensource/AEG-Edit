from dataclasses import dataclass
from typing import List, Optional

from util.hparams import HyperParams


@dataclass
class AGraceHyperParams(HyperParams):
    """A-GRACE Hyperparameter configuration"""
    
    # Model info
    model_name: str
    tokenizer_name: str
    
    # Module templates
    inner_params: List[str]
    
    # Training settings
    edit_lr: float
    n_iter: int
    
    # A-GRACE specific params
    eps: float  # Initial epsilon for key matching
    dist_fn: str  # Distance function: euc, mmd, cos
    val_init: str  # Value initialization: cold, warm
    val_train: str  # Value training: sgd, pert
    val_reg: str  # Value regularization
    reg: str  # Regularization method
    replacement: str  # Replacement strategy: replace_last, replace_all, replace_prompt
    eps_expand: str  # Epsilon expansion: coverage, moving_average
    num_pert: int  # Number of perturbations
    dropout: float
    
    encoder_state_path: str
    
    # Device
    device: int = 0
    
    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False

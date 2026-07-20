"""
AlphaEdit_ARE_AEG: AlphaEdit-ARE with AEG-Edit target construction.

The underlying editing logic is unchanged. This package only standardizes the
public naming around:

1. API-weighted windowing
2. dual-view graph encoding
3. graph-to-hidden alignment
"""

from .AlphaEdit_ARE_AEG_main import apply_alphaedit_are_aeg_to_model, get_cov
from .AlphaEdit_ARE_AEG_hparams import AlphaEditAREAEGHyperParams
from .compute_aeg_edit_targets import (
    compute_aeg_edit_targets,
    AEGEncoder,
    ApiEvolutionGraphRGCNLayer,
    ApiEvolutionGraphEncoder,
    GraphAlignmentAdapter,
    GraphToHiddenAlignmentAdapter,
    build_api_weighted_windows,
)

__all__ = [
    "apply_alphaedit_are_aeg_to_model",
    "get_cov",
    "AlphaEditAREAEGHyperParams",
    "compute_aeg_edit_targets",
    "AEGEncoder",
    "ApiEvolutionGraphRGCNLayer",
    "ApiEvolutionGraphEncoder",
    "GraphAlignmentAdapter",
    "GraphToHiddenAlignmentAdapter",
    "build_api_weighted_windows",
]

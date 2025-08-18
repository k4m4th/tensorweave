"""
tensorweave
===========

Top-level package API re-export.

This file re-exports the main classes/functions from:
- tensorweave.tensorweave (your core model/ops)
- tensorweave.utils       (plotting, sampling, io, etc.)

"""

# Core model & autodiff utilities (import the things you want public)
from .tensorweave import (
    NeuralFourierField,
    RFFEnsemble,
    compute_gradient,
    compute_hessian,
    compute_hessian_eval,
)

# Utils (plotting, sampling, preprocessing)
from .utils import (
    hist_equalize,
    poisson_disk_indices,
    poisson_disk_indices_naive,
    add_ftg_noise_by_snr,
    filter_by_min_distance,
    average_within_radius,
    read_and_subsample_lines,
    plot_hessian_components,
    spectral_integration,
)

__all__ = [
    # core
    "NeuralFourierField",
    "RFFEnsemble",
    "compute_gradient",
    "compute_hessian",
    "compute_hessian_eval",
    # utils
    "hist_equalize",
    "poisson_disk_indices",
    "poisson_disk_indices_naive",
    "add_ftg_noise_by_snr",
    "filter_by_min_distance",
    "average_within_radius",
    "read_and_subsample_lines",
    "plot_hessian_components",
    "spectral_integration",
    # aliases
    "poisson_disk_grid",
    "plot2D",
]
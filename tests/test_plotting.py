import numpy as np
import torch
import tensorweave as tw


def test_plot_hessian_components_runs():
    # Create a small grid and a fake Hessian on all active cells
    nx, ny = 10, 12
    x = np.linspace(0, 9, nx)
    y = np.linspace(0, 11, ny)
    N = nx * ny
    active = np.ones(N, dtype=bool)

    # Constant Hessian (diagonal)
    h = torch.zeros(N, 3, 3)
    h[:, 0, 0] = 2.0
    h[:, 1, 1] = 1.0
    h[:, 2, 2] = 0.5

    # Should not raise
    tw.plot_hessian_components(
        predicted_hessian=h,
        active_mask=active,
        x_coords=x,
        y_coords=y,
        components=("xx", "xy", "xz"),
        cmap="Spectral",
        equalize=True,
    )
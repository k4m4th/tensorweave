import numpy as np
import torch
import torch.nn as nn
import tensorweave as tw


def _synthetic_quadratic(N=200, bounds=200.0):
    """
    Quadratic potential: phi = 0.5*(ax*x^2 + ay*y^2 + az*z^2)
    Hessian = diag(ax, ay, az), off-diagonals zero.
    """
    ax, ay, az = 2.0, 1.5, 1.0
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-bounds, bounds, size=(N, 3)).astype(np.float32)

    # 6 independent components in order [xx, xy, xz, yy, yz, zz]
    h6 = np.tile(np.array([ax, 0.0, 0.0, ay, 0.0, az], dtype=np.float32), (N, 1))
    return xyz, h6


def test_fit_reduces_loss_and_predicts_shapes():
    # Data
    xyz, ftg6 = _synthetic_quadratic(N=300, bounds=50.0)

    # Grid for Laplacian sampling (use the same coords for speed)
    grid = xyz.copy()

    # Tensors
    coords_t = torch.tensor(xyz, dtype=torch.float32)
    data_t = torch.tensor(ftg6, dtype=torch.float32)
    grid_t = torch.tensor(grid, dtype=torch.float32)

    # Model (small & stable)
    model = tw.NeuralFourierField(
        input_dim=3,
        num_fourier_features=16,
        length_scales=[10.0, 100.0],
        learnable=False,           # keep LS fixed for a deterministic test
        hidden_layers=[64],
        activation=nn.SiLU(),
        potential_scale=1.0,
        harmonic=False,
        device="cpu",
        seed=123,
    )

    losses, lap_counts = model.fit(
        coords=coords_t,
        data=data_t,
        grid=grid_t,
        patience=50,  # early stopping patience
        min_delta=0.002,
        epochs=500,            # short but enough to see drop
        lr=1e-3,
        lap_spacing=25.0,
        lap_samples=500,
        chunk_size=128,
        loss_fn=nn.L1Loss(),
        plot_every=0,
    )

    # Loss should drop on average
    assert losses.shape[0] >= 10
    assert losses.shape[1] == 7 # 6 hessian components + laplacian loss
    start = float(losses[0].sum())
    end = float(losses[-1].sum())
    assert end < start

    # Prediction API sanity
    phi = model.predict(xyz, output="potential")
    grad = model.predict(xyz, output="gradient")
    hess = model.predict(xyz, output="hessian")
    assert phi.shape == (xyz.shape[0], 1)
    assert grad.shape == (xyz.shape[0], 3)
    assert hess.shape == (xyz.shape[0], 3, 3)

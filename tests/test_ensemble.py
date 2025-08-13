import numpy as np
import torch
import torch.nn as nn
import tensorweave as tw


def test_ensemble_predict_shapes():
    # tiny synthetic dataset
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-50, 50, size=(200, 3)).astype(np.float32)
    # Simple target: diagonal Hessian (2.0, 1.0, 0.5)
    ftg6 = np.tile(np.array([2.0, 0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32), (xyz.shape[0], 1))
    grid = xyz.copy()

    coords_t = torch.tensor(xyz, dtype=torch.float32)
    data_t = torch.tensor(ftg6, dtype=torch.float32)
    grid_t = torch.tensor(grid, dtype=torch.float32)

    model_kwargs = dict(
        input_dim=3,
        num_fourier_features=12,
        length_scales=[40.0, 120.0],
        learnable=False,
        hidden_layers=[64],
        activation=nn.SiLU(),
        potential_scale=1.0,
        harmonic=False,
        device="cpu",
    )

    ens = tw.RFFEnsemble(n_members=2, model_kwargs=model_kwargs, base_seed=7, bootstrap=False, keep_members=True)
    ens.fit(
        coords_t, data_t, grid_t,
        epochs=20, lr=3e-4, lap_spacing=20.0, lap_samples=300, chunk_size=128, loss_fn=nn.L1Loss()
    )

    mean, std, qpair = ens.predict(xyz, output="hessian", return_std=True, return_quantiles=(0.1, 0.9))
    assert mean.shape == (xyz.shape[0], 3, 3)
    assert std.shape == (xyz.shape[0], 3, 3)
    assert isinstance(qpair, tuple) and len(qpair) == 2
    assert qpair[0].shape == mean.shape and qpair[1].shape == mean.shape

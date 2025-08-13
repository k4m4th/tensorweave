import numpy as np
import torch
import torch.nn as nn
import tensorweave as tw


def test_forward_output_shape_cpu():
    model = tw.NeuralFourierField(
        input_dim=3,
        num_fourier_features=16,
        length_scales=[200.0, 500.0],
        learnable=False,
        hidden_layers=[64],
        activation=nn.SiLU(),
        potential_scale=1.0,
        harmonic=False,
        device="cpu",
        seed=7,
    )
    x = torch.randn(37, 3)
    y = model(x)
    assert y.shape == (37, 1)


def test_predict_shapes_numpy():
    model = tw.NeuralFourierField(
        input_dim=3,
        num_fourier_features=8,
        length_scales=[100.0, 400.0],
        learnable=False,
        hidden_layers=[32],
        activation=nn.SiLU(),
        potential_scale=1.0,
        harmonic=False,
        device="cpu",
        seed=11,
    )

    coords = np.random.randn(25, 3).astype(np.float32)
    phi = model.predict(coords, output="potential")
    grad = model.predict(coords, output="gradient")
    hess = model.predict(coords, output="hessian")

    assert phi.shape == (25, 1)
    assert grad.shape == (25, 3)
    assert hess.shape == (25, 3, 3)
import torch
import tensorweave as tw
import torch.nn as nn


def test_compute_gradient_and_hessian_shapes():
    model = tw.NeuralFourierField(
        input_dim=3,
        num_fourier_features=12,
        length_scales=[150.0, 600.0],
        learnable=False,
        hidden_layers=[64],
        activation=nn.SiLU(),
        potential_scale=1.0,
        harmonic=False,
        device="cpu",
        seed=3,
    )
    coords = torch.randn(40, 3)
    phi, grad = tw.compute_gradient(model, coords, normalize=False)
    assert phi.shape == (40, 1)
    assert grad.shape == (40, 3)

    phi2, grad2, hess = tw.compute_hessian(model, coords, chunk_size=16, device="cpu")
    assert phi2.shape == (40, 1)
    assert grad2.shape == (40, 3)
    assert hess.shape == (40, 3, 3)

    phi3, grad3, hess3 = tw.compute_hessian_eval(model, coords, chunk_size=16, device="cpu")
    assert hess3.shape == (40, 3, 3)
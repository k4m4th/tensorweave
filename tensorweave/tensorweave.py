"""
Neural Fourier Field (RFF) module for scalar-potential interpolation and
Hessian-based tensor extraction (e.g., FTG components).

This module provides:
- A neural field model (`NeuralFourierField`) with random Fourier feature (RFF)
  encoding across multiple length scales, optional harmonic decay in z,
  and a simple MLP head.
- Utilities for first- and second-order autodiff (gradient & Hessian).
- A training loop (`fit`) with Poisson-disk sampling for global Laplacian
  regularization and a clean, callback-based plotting interface.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Helper functions
from .utils import poisson_disk_indices


# --------------------------------------------------------------------------- #
#                             Model Definition                                 #
# --------------------------------------------------------------------------- #

class NeuralFourierField(nn.Module):
    """
    Neural field with Random Fourier Features (RFF) for scalar potential modeling.

    The model maps input coordinates :math:`\\mathbf{x} = (x, y, z)` to a scalar
    potential :math:`\\phi(\\mathbf{x})`. From this potential, first and second
    derivatives (gradient and Hessian) can be obtained via autodiff. In many
    geophysical use-cases, elements of the Hessian correspond to gravity
    gradiometry components.

    Pipeline
    --------
    1. **RFF Encoding**: The in-plane coordinates (all but the last dim) are
       projected onto random spatial frequencies at multiple length scales,
       producing sinusoidal features. If ``harmonic=True``, each feature is
       multiplied by an exponential decay term in the last dimension (e.g., z).
    2. **MLP Head**: Encoded features are fed to a small MLP to predict the
       scalar potential (scaled by a learnable or fixed factor in log10 space).
    3. **Autodiff**: Use provided helpers to obtain gradient/Hessian.

    Notes
    -----
    - Frequency vectors are sampled once at instantiation (seed-controlled).
    - Length scales and potential scale are stored in base-10 logarithmic space
      for numerical stability.
    - The model is device-aware and places all parameters/tensors on ``device``.

    Parameters
    ----------
    input_dim : int, default=3
        Dimensionality of input coordinates. Commonly 3 for (x, y, z).
    num_fourier_features : int, default=32
        Number of Fourier features per length scale (for each of cos/sin).
    harmonic : bool, default=True
        If True, apply an exponential decay in the last coordinate dimension
        proportional to the feature wavenumber norm, i.e.,
        :math:`\\exp(-z \\lVert \\mathbf{k}\\rVert)`.
    distribution : {"Normal", "Cauchy"}, default="Normal"
        Distribution of random spatial frequencies.
    dist_variance : float | str, default="1"
        Scale parameter for the Cauchy distribution (kept for parity with
        prior code). Strings are cast to float.
    potential_scale : float, default=1e2
        Output scale for the potential (stored/learned in log10 space).
    length_scales : list[float], default=[1e2, 5e2, 1e3]
        Length scales used to scale the frequency vectors (stored/learned in
        log10 space). Larger scales encourage longer wavelengths.
    learnable : bool, default=True
        If True, ``length_scales`` and ``potential_scale`` are learnable
        parameters; otherwise they are fixed buffers.
    hidden_layers : list[int], default=[512]
        Hidden layer sizes for the MLP head.
    output_dim : int, default=1
        Output dimension of the MLP; typically 1 for scalar potential.
    activation : nn.Module, default=nn.SiLU()
        Activation function used after each linear layer (except the last).
    seed : int, default=404
        RNG seed for frequency sampling (reproducibility).
    device : str | None, default=None
        Torch device string. If None, uses "cuda" if available else "cpu".

    Attributes
    ----------
    device : str
        Device in use by the module.
    k_xy : torch.Tensor
        Random spatial frequencies (shape: ``(input_dim-1, num_fourier_features)``).
    k_z_norm : torch.Tensor
        Norm of frequency vectors (shape: ``(1, num_fourier_features)``), used
        for harmonic decay.
    potential_scale : torch.Tensor | nn.Parameter
        Log10 of the potential scale.
    length_scales : torch.Tensor | nn.Parameter
        Log10 of length scales :math:`\\log_{10}(\\ell)`.
    mlp : nn.Sequential
        The MLP head.

    Examples
    --------
    >>> model = NeuralFourierField(
    ...     input_dim=3, num_fourier_features=64, length_scales=[200., 500., 1000.]
    ... )
    >>> coords = torch.randn(1000, 3)
    >>> phi = model(coords)  # (1000, 1)
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_fourier_features: int = 32,
        harmonic: bool = True,
        distribution: str = "Normal",
        dist_variance: float | str = "1",
        potential_scale: float = 1e2,
        length_scales: List[float] = [1e2, 5e2, 1e3],
        learnable: bool = True,
        hidden_layers: List[int] = [512],
        output_dim: int = 1,
        activation: nn.Module = nn.SiLU(),
        seed: int = 404,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        # ------------------------------ device -------------------------------- #
        self.device: str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------ config -------------------------------- #
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.harmonic: bool = harmonic
        self.use_rff: bool = num_fourier_features > 0
        self.activation: nn.Module = activation

        # ---------------------------- RFF setup ------------------------------- #
        if self.use_rff:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)

            if distribution == "Normal":
                self.k_xy = torch.randn(
                    input_dim - 1, num_fourier_features, device=self.device, generator=gen
                )
            elif distribution == "Cauchy":
                # Preserve prior behavior and accept string scale
                scale = float(dist_variance) if isinstance(dist_variance, str) else dist_variance
                self.k_xy = torch.distributions.Cauchy(0, scale).sample(
                    (input_dim - 1, num_fourier_features)
                ).to(self.device)
            else:
                raise ValueError('distribution must be one of {"Normal", "Cauchy"}')

            # Norm for harmonic decay along last coordinate (e.g., z)
            self.k_z_norm = torch.norm(self.k_xy, dim=0, keepdim=True)  # (1, M)

            # Store (and optionally learn) parameters in log10 space
            if learnable:
                self.potential_scale = nn.Parameter(
                    torch.log10(torch.tensor(potential_scale, dtype=torch.float32, device=self.device))
                )
                self.length_scales = nn.Parameter(
                    torch.log10(torch.tensor(length_scales, dtype=torch.float32, device=self.device))
                )
            else:
                self.register_buffer(
                    "potential_scale",
                    torch.log10(torch.tensor(potential_scale, dtype=torch.float32, device=self.device)),
                )
                self.register_buffer(
                    "length_scales",
                    torch.log10(torch.tensor(length_scales, dtype=torch.float32, device=self.device)),
                )

        # ------------------------------ MLP ----------------------------------- #
        mlp_input_dim = 2 * num_fourier_features * len(length_scales) if self.use_rff else input_dim
        layer_dims = [mlp_input_dim] + hidden_layers + [output_dim]

        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(layer_dims[:-2], layer_dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim, device=self.device))
            if self.activation is not None:
                layers.append(self.activation)
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1], device=self.device))
        self.mlp = nn.Sequential(*layers)

        # Xavier init for linear layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        self.to(self.device)

    # --------------------------- public interface ---------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode (if enabled) and infer scalar potential.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar potential with shape ``(N, output_dim)``.
        """
        x = x.to(self.device)
        features = self._encode_rff(x) if self.use_rff else x
        # potential scale is kept in log10 space
        return self.mlp(features) * (10 ** self.potential_scale)

    # ----------------------------- rff encoder ------------------------------- #

    def _encode_rff(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Random Fourier Feature (RFF) encoder with optional harmonic decay.

        Projects in-plane coordinates (all but last dim) onto random frequencies
        at each length scale and concatenates cos/sin features. If ``harmonic``,
        multiplies by :math:`\\exp(-z \\lVert \\mathbf{k}\\rVert)`.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded features of shape ``(N, 2 * M * n_scales)`` where
            ``M = num_fourier_features``.
        """
        xy = coords[:, :-1]                            # (N, D-1) -> usually (N, 2)
        z_col = coords[:, -1].unsqueeze(-1)            # (N, 1)

        blocks: List[torch.Tensor] = []
        num_scales = int(self.length_scales.numel())

        for i in range(num_scales):
            scale = 10 ** (self.length_scales[i])      # recover linear scale
            k_eff = self.k_xy / scale                  # (D-1, M)
            proj = xy @ k_eff                          # (N, M)

            if self.harmonic:
                k_norm = torch.norm(k_eff, dim=0, keepdim=True)  # (1, M)
                decay = torch.exp(-z_col @ k_norm)               # (N, M)
            else:
                decay = 1.0

            cos_features = torch.cos(proj) * decay
            sin_features = torch.sin(proj) * decay
            blocks.append(torch.cat([cos_features, sin_features], dim=-1))

        return torch.cat(blocks, dim=-1)

    # ------------------------------- training -------------------------------- #

    def fit(
        self,
        coords: torch.Tensor,
        data: torch.Tensor,
        grid: torch.Tensor,
        epochs: int = 250,
        loss_fn: nn.Module = nn.L1Loss(),
        patience: int = 50,
        lr: float = 1e-4,
        lap_spacing: float | Tuple[float, float, float, int, int] = 100.0,
        lap_samples: int = 2000,
        chunk_size: int = 512,
        # --- plotting during training ----# 
        plot_every: int = 0,
        plotter: Optional[Callable[[torch.Tensor], None]] = None,
        eval_grid: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Train the field on known FTG components with Laplacian regularization.

        This routine matches selected Hessian components at provided coordinates
        and regularizes the global solution by penalizing the Laplacian (trace
        of the Hessian) at Poisson-disk-sampled grid points.

        Losses
        ------
        - **Data Loss**: pointwise loss (e.g., L1) between selected Hessian
          components and provided ``data`` for known points. We use the six
          independent components ``[Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]`` picked from
          the flattened 3x3 Hessian.
        - **Laplacian Loss**: mean absolute value of :math:`\\mathrm{tr}(H)` on
          Poisson-disk samples.

        Dynamic weighting balances these two losses each epoch based on their
        relative magnitudes.

        Poisson-Disk Radius Scheduling
        ------------------------------
        ``lap_spacing`` may be either:
        - ``float``: fixed Poisson-disk radius.
        - ``(r_max, r_min, decay, cycles, update_schedule)``:
            a simple exponential schedule that cycles every
            ``epochs/cycles`` epochs, interpolating from ``r_max``
            to ``r_min`` with exponential decay factor ``decay``.
            (Behavior preserved for parity with earlier code.) r is
            updated once every ``update_schedule`` epochs.

        Plotting
        --------
        To keep training clean, plotting is callback-based. Provide:
        - ``plotter(pred_hessian: torch.Tensor) -> None``: a function that takes
          the predicted Hessian on ``eval_grid`` (or ``grid`` if not provided).
        - ``plot_every``: plot every N epochs (set 0 to disable).

        Parameters
        ----------
        coords : torch.Tensor
            Known coordinates of shape ``(N, input_dim)``.
        data : torch.Tensor
            Target FTG components at known points of shape ``(N, 6)`` in the
            order ``[Gxx, Gxy, Gxz, Gyy, Gyz, Gzz]``.
        grid : torch.Tensor
            Candidate grid of shape ``(M, input_dim)`` used to sample Laplacian points.
        epochs : int, default=250
            Maximum training epochs.
        loss_fn : nn.Module, default=nn.L1Loss()
            Pointwise loss function for FTG matching.
        patience : int, default=50
            Early-stopping patience on the unweighted loss (data + laplacian).
        lr : float, default=1e-4
            Learning rate for Adam.
        lap_spacing : float | tuple, default=100.0
            Poisson-disk radius or scheduling tuple as described above.
        lap_samples : int, default=2000
            Maximum number of Poisson-disk samples per epoch.
        chunk_size : int, default=512
            Chunk size for Hessian computations.
        plot_every : int, default=0
            Plot every N epochs (0 disables plotting).
        plotter : callable | None, default=None
            Callback that receives predicted Hessian on ``eval_grid`` (or ``grid``).
        eval_grid : torch.Tensor | None, default=None
            Grid to evaluate for plotting; falls back to ``grid`` if None.

        Returns
        -------
        (np.ndarray, list[int])
            - Array of shape ``(epochs_eff, 2)`` with columns
              ``[data_loss, laplacian_loss]`` per epoch.
            - List of Laplacian sample counts per epoch.

        Notes
        -----
        - Learning-rate scheduling uses ``ReduceLROnPlateau`` on the unweighted
          sum of losses (data + laplacian).
        - Early stopping also monitors this unweighted sum.
        """
        self.train()
        coords = coords.to(self.device)
        data = data.to(self.device)
        grid = grid.to(self.device)
        eval_grid = (eval_grid if eval_grid is not None else grid).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)

        best_loss = float("inf")
        best_state = copy.deepcopy(self.state_dict())
        no_improve_count = 0

        # Pre-compute arrays for Poisson-disk sampling (on CPU as required)
        grid_np = grid.detach().cpu().numpy()
        x_vec, y_vec = grid_np[:, 0], grid_np[:, 1]

        self.data_loss_history: List[float] = []
        self.laplacian_loss_history: List[float] = []
        self.laplacian_sample_counts: List[int] = []

        with tqdm(range(epochs), desc="Training") as bar:
            for epoch in bar:
                optimizer.zero_grad(set_to_none=True)

                # ---- Determine Poisson-disk radius r ---- #
                if isinstance(lap_spacing, tuple):
                    r_max, r_min, decay, cycles, update_schedule = lap_spacing
                    cycle_len = max(1, int(epochs / max(1, int(cycles))))
                    # Same periodic exponential schedule as before (epoch % 20 gate kept)
                    if epoch % update_schedule == 0:
                        phase = (epoch % cycle_len) / max(1, cycle_len)
                        r = r_min + (r_max - r_min) * np.exp(-decay * phase)
                else:
                    r = float(lap_spacing)

                # ---- Poisson-disk sample subset for Laplacian penalty ---- #
                lap_indices = poisson_disk_indices(x_vec, y_vec, r, lap_samples)
                lap_sample = grid[lap_indices]
                self.laplacian_sample_counts.append(int(lap_sample.shape[0]))

                # ---- Compute Hessian on concatenated coords (known ⊕ lap) ---- #
                all_coords = torch.cat([coords, lap_sample], dim=0)
                _, _, all_hessians = compute_hessian(self, all_coords,
                                                     chunk_size=chunk_size,
                                                     device=self.device)

                # ---- Data loss on known points ---- #
                known_h = all_hessians[: coords.shape[0]]               # (N, 3, 3)
                # Flatten 3x3 -> 9 with row-major order:
                # [xx, xy, xz, yx, yy, yz, zx, zy, zz]
                # Select 6 independent components: [xx, xy, xz, yy, yz, zz]
                known_flat = known_h.reshape(-1, 9)[:, (0, 1, 2, 4, 5, 8)]
                data_loss = loss_fn(known_flat, data)
                self.data_loss_history.append(float(data_loss.detach().cpu()))

                # ---- Laplacian loss on lap points ---- #
                lap_h = all_hessians[coords.shape[0] :]
                # trace(H) via diagonal contraction
                laplacian = torch.einsum("...ii->...", lap_h)
                laplacian_loss = torch.mean(torch.abs(laplacian))
                self.laplacian_loss_history.append(float(laplacian_loss.detach().cpu()))

                # ---- Unweighted loss for scheduling/early-stopping ---- #
                current_loss = data_loss + laplacian_loss

                # ---- Dynamic weighting (normalized by sum) ---- #
                denom = float(current_loss.detach().cpu()) + 1e-12
                w_data = float(data_loss.detach().cpu()) / denom
                w_lap = float(laplacian_loss.detach().cpu()) / denom
                total_loss = data_loss * w_data + laplacian_loss * w_lap

                # ---- Backprop, step, schedule ---- #
                total_loss.backward()
                optimizer.step()
                scheduler.step(current_loss.cpu().detach())

                # ---- Early stopping ---- #
                curr = float(current_loss.cpu().detach())
                if curr < best_loss:
                    best_loss = curr
                    best_state = copy.deepcopy(self.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"No improvement for {patience} epochs. Early stopping.")
                        break

                # ---- Progress bar ---- #
                bar.set_postfix(
                    {
                        "loss": f"{curr:.6f}",
                        "data": f"{self.data_loss_history[-1]:.6f}",
                        "lap": f"{self.laplacian_loss_history[-1]:.6f}",
                        "lr": optimizer.param_groups[0]["lr"],
                        "lap_ns": str(self.laplacian_sample_counts[-1]),
                        "stall": no_improve_count,
                    }
                )

                # ---- Optional plotting ---- #
                if plotter is not None and plot_every > 0 and (epoch % plot_every == 0):
                    # compute_hessian_eval builds second derivatives without keeping
                    # third-order graphs and detaches CPU copies internally.
                    _, _, pred_hessian = compute_hessian_eval(
                        self, eval_grid, chunk_size=max(1024, chunk_size), device=self.device
                    )
                    plotter(pred_hessian)

        # Restore best model parameters
        self.load_state_dict(best_state)

        # Return history arrays (N_epochs_eff x 2) and sample counts
        losses = np.c_[self.data_loss_history, self.laplacian_loss_history]
        return losses, self.laplacian_sample_counts
    
    # ------------------------------- prediction -------------------------------- #
    
    def predict(
        self,
        coords: np.ndarray,
        output: str = "hessian",
        chunk_size: int = 2048,
        normalize_grad: bool = False,
    ) -> np.ndarray:
        """
        Predict quantities from coordinates, sklearn-style.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape ``(N, input_dim)`` with input coordinates.
        output : {"potential", "gradient", "hessian"}, default="potential"
            Which quantity to return:
            - "potential": scalar potential :math:`\\phi(\\mathbf{x})`, shape ``(N, 1)``.
            - "gradient":  first derivative :math:`\\nabla\\phi`, shape ``(N, D)``.
            - "hessian":   full Hessian :math:`\\nabla^2\\phi`, shape ``(N, D, D)``.
            Synonyms accepted: {"phi", "grad", "hess"}.
        chunk_size : int, default=2048
            Number of points per chunk to control memory use.
        normalize_grad : bool, default=False
            If ``True`` and ``output="gradient"``, L2-normalize gradient vectors.

        Returns
        -------
        np.ndarray
            The requested quantity as a NumPy array on CPU. Shapes as described above.

        Notes
        -----
        - Uses efficient evaluation paths per quantity:
          * Potential: forward-only (no autograd graph).
          * Gradient: first derivative w.r.t. inputs (no higher-order graph).
          * Hessian: uses ``compute_hessian_eval`` (second derivatives, eval-safe).
        - The model's train/eval mode is restored after prediction.
        """
        # ------------- validate input ------------- #
        if not isinstance(coords, np.ndarray):
            raise TypeError("coords must be a NumPy array of shape (N, input_dim).")
        if coords.ndim != 2 or coords.shape[1] != self.input_dim:
            raise ValueError(
                f"coords must have shape (N, {self.input_dim}), got {coords.shape}."
            )

        # Normalize output keyword
        key = output.strip().lower()
        if key in {"phi"}:
            key = "potential"
        elif key in {"grad"}:
            key = "gradient"
        elif key in {"hess"}:
            key = "hessian"
        if key not in {"potential", "gradient", "hessian"}:
            raise ValueError('output must be one of {"potential", "gradient", "hessian"}.')

        # Preserve current mode, switch to eval for inference
        prior_mode = self.training
        self.eval()
        try:
            # ------- convert input once (no copy if already float32 C-contig) ------- #
            # We'll slice into chunks below (to keep peak memory low).
            n = coords.shape[0]
            out_list: list[torch.Tensor] = []

            if key == "potential":
                # Forward-only path: no autograd
                with torch.no_grad():
                    for start in range(0, n, chunk_size):
                        end = start + chunk_size
                        x_chunk = torch.from_numpy(coords[start:end]).to(
                            self.device, dtype=torch.float32, non_blocking=True
                        )
                        y_chunk = self.forward(x_chunk)  # (B, 1)
                        out_list.append(y_chunk.detach().cpu())
                y = torch.cat(out_list, dim=0)  # (N, 1)
                return y.numpy()

            elif key == "gradient":
                # Need first derivative w.r.t inputs, but no higher-order graph
                for start in range(0, n, chunk_size):
                    end = start + chunk_size
                    x_chunk = torch.from_numpy(coords[start:end]).to(
                        self.device, dtype=torch.float32, non_blocking=True
                    ).requires_grad_(True)

                    # Enable grads for inputs only
                    with torch.enable_grad():
                        phi = self.forward(x_chunk)
                        grad = torch.autograd.grad(
                            outputs=phi,
                            inputs=x_chunk,
                            grad_outputs=torch.ones_like(phi),
                            create_graph=False,  # we don't need Hessian here
                            retain_graph=False,
                            allow_unused=False,
                        )[0]

                        if normalize_grad:
                            norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
                            grad = grad / norm

                    out_list.append(grad.detach().cpu())
                g = torch.cat(out_list, dim=0)  # (N, D)
                return g.numpy()

            else:  # key == "hessian"
                # Reuse the eval-safe Hessian helper (already chunked & detached)
                coords_t = torch.from_numpy(coords).to(self.device, dtype=torch.float32)
                _, _, hess = compute_hessian_eval(
                    self, coords_t, chunk_size=chunk_size, device=self.device
                )
                # compute_hessian_eval returns CPU tensors already; just to be safe:
                return hess.detach().cpu().numpy()

        finally:
            # Restore prior training mode
            self.train(prior_mode)
            
# --------------------------------------------------------------------------- #
#                                  Ensemble                                   #
# --------------------------------------------------------------------------- #

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class RFFEnsemble:
    """
    Deep ensemble of NeuralFourierField models.

    Train K independent members, each with different seeds (hence different RFF
    encodings and MLP initializations), optionally with bootstrapped data
    subsets. Aggregate predictions for mean/uncertainty.

    Parameters
    ----------
    n_members : int
        Number of ensemble members.
    model_kwargs : dict
        Keyword args passed to `NeuralFourierField(**model_kwargs, seed=...)`.
        Do not include `seed` here; it is set per-member automatically.
    base_seed : int, default=404
        Base seed; each member uses `base_seed + i`.
    bootstrap : bool, default=True
        If True, sample with replacement per member's training set.
    bagging_frac : float, default=1.0
        Fraction of the training set size to draw for each bootstrapped sample.
        Ignored if `bootstrap=False`.
    device : str | None, default=None
        Device override for model instantiation; if None, the model decides.
    keep_members : bool, default=True
        If True, keep fitted models for later reuse. If False, frees them after fitting.
    """
    n_members: int
    model_kwargs: Dict
    base_seed: int = 404
    bootstrap: bool = True
    bagging_frac: float = 1.0
    device: Optional[str] = None
    keep_members: bool = True

    members: List[NeuralFourierField] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    histories: List[Tuple[np.ndarray, List[int]]] = field(default_factory=list)

    # ------------- generate a NeuralFourierField ------------- #
    def _make_member(self, seed: int) -> NeuralFourierField:
        kwargs = dict(self.model_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device
        kwargs["seed"] = seed
        return NeuralFourierField(**kwargs)
    
    # ------------- ensemble fitting ------------- #
    def fit(
        self,
        coords: torch.Tensor,
        data: torch.Tensor,
        grid: torch.Tensor,
        *,
        # pass-through to model.fit
        epochs: int = 250,
        loss_fn: nn.Module = nn.L1Loss(),
        patience: int = 50,
        lr: float = 1e-4,
        lap_spacing: float | Tuple[float, float, float, int] = 100.0,
        lap_samples: int = 2000,
        chunk_size: int = 512,
        plot_every: int = 0,
        plotter: Optional[Callable[[torch.Tensor], None]] = None,
        eval_grid: Optional[torch.Tensor] = None,
        # reproducibility of bootstrapping
        bootstrap_seed: Optional[int] = None,
    ) -> "RFFEnsemble":
        """
        Fit all ensemble members.

        Each member is trained on either the full dataset (bootstrap=False) or a
        bootstrapped subset (with replacement) of size `bagging_frac * N`.

        Returns
        -------
        self : RFFEnsemble
        """
        N = coords.shape[0]
        indices_all = np.arange(N)

        if bootstrap_seed is not None:
            rng = np.random.default_rng(bootstrap_seed)
        else:
            rng = np.random.default_rng(self.base_seed + 12345)

        self.members.clear()
        self.seeds = [self.base_seed + i for i in range(self.n_members)]
        self.histories.clear()

        for i, seed in enumerate(self.seeds):
            # Print the Model Number
            print("Model #" + str(i))
            # different seeds -> different RFFs & MLP inits
            _set_all_seeds(seed)
            member = self._make_member(seed)

            # choose training indices for this member
            if self.bootstrap:
                m = max(1, int(self.bagging_frac * N))
                boot_idx = rng.choice(indices_all, size=m, replace=True)
            else:
                boot_idx = indices_all  # full data

            coords_i = coords[boot_idx]
            data_i = data[boot_idx]

            # optional: enforce identical Poisson sampling across ensemble by
            # fixing numpy's global RNG here; comment out to diversify
            np.random.seed(seed)

            history = member.fit(
                coords=coords_i,
                data=data_i,
                grid=grid,
                epochs=epochs,
                loss_fn=loss_fn,
                patience=patience,
                lr=lr,
                lap_spacing=lap_spacing,
                lap_samples=lap_samples,
                chunk_size=chunk_size,
                plot_every=plot_every if i == 0 else 0,  # plot only first member by default
                plotter=plotter if i == 0 else None,
                eval_grid=eval_grid,
            )
            self.histories.append(history)
            if self.keep_members:
                self.members.append(member)
            else:
                # free if not keeping
                del member
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    # ------------- ensemble predictions ------------- #                
    def predict(
        self,
        coords: np.ndarray,
        output: str = "hessian",
        chunk_size: int = 2048,
        normalize_grad: bool = False,
        return_std: bool = True,
        return_quantiles: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Aggregate predictions from the ensemble.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of shape (N, D).
        output : {"potential","gradient","hessian"}, default="hessian"
            Quantity to predict.
        chunk_size : int, default=2048
            Chunk size for each member's prediction.
        normalize_grad : bool, default=False
            Normalize gradients if output="gradient".
        return_std : bool, default=True
            If True, also return the ensemble standard deviation.
        return_quantiles : (float, float) | None, default=None
            If set, e.g., (0.05, 0.95), also return lower/upper quantiles.

        Returns
        -------
        mean : np.ndarray
            Ensemble mean prediction, shape depends on `output`:
            - potential: (N, 1), gradient: (N, D), hessian: (N, D, D)
        std : np.ndarray | None
            Ensemble std over members (same shape as mean) if `return_std=True`.
        (q_lo, q_hi) : tuple[np.ndarray, np.ndarray] | None
            Requested quantiles if `return_quantiles` is provided.
        """
        if not self.members:
            raise RuntimeError("Ensemble has no trained members. Call fit() first.")

        preds = []
        for m in self.members:
            y = m.predict(
                coords=coords,
                output=output,
                chunk_size=chunk_size,
                normalize_grad=normalize_grad,
            )
            preds.append(y)

        stack = np.stack(preds, axis=0)  # (K, ...)  K = n_members
        mean = stack.mean(axis=0)

        std = stack.std(axis=0, ddof=1) if return_std else None

        qpair = None
        if return_quantiles is not None:
            qlo, qhi = return_quantiles
            q_low = np.quantile(stack, qlo, axis=0)
            q_high = np.quantile(stack, qhi, axis=0)
            qpair = (q_low, q_high)

        return mean, std, qpair

# --------------------------------------------------------------------------- #
#                    Autodiff: Gradient & Hessian Utilities                   #
# --------------------------------------------------------------------------- #

def compute_gradient(
    model: nn.Module,
    coords: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scalar potential and its gradient at provided coordinates.

    Parameters
    ----------
    model : nn.Module
        Model mapping coordinates -> scalar potential.
    coords : torch.Tensor
        Input coordinates of shape ``(N, D)``.
    normalize : bool, default=True
        If True, L2-normalize gradient vectors (with epsilon).

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        - Scalar potential of shape ``(N, 1)``.
        - Gradient field of shape ``(N, D)``.
    """
    coords = coords.to(getattr(model, "device", coords.device)).requires_grad_(True)
    scalar_field = model(coords)
    grad_field = torch.autograd.grad(
        outputs=scalar_field,
        inputs=coords,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
    )[0]

    if normalize:
        norm = torch.norm(grad_field, dim=-1, keepdim=True) + 1e-8
        grad_field = grad_field / norm

    return scalar_field, grad_field


def compute_hessian(
    model: nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 1024,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute scalar field, gradient, and full Hessian in chunks.

    Parameters
    ----------
    model : nn.Module
        Model mapping coordinates -> scalar potential.
    coords : torch.Tensor
        Input coordinates of shape ``(N, D)``.
    chunk_size : int, default=1024
        Chunk size for memory control.
    device : str | None, default=None
        Explicit device. Defaults to ``model.device`` if available.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        - Scalar field of shape ``(N, 1)``.
        - Gradient field of shape ``(N, D)``.
        - Hessian tensor of shape ``(N, D, D)``.
    """
    target_device = device if device is not None else getattr(model, "device", coords.device)
    coords = coords.to(target_device)

    scalar_fields: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    hessians: List[torch.Tensor] = []

    for start in range(0, coords.shape[0], chunk_size):
        end = start + chunk_size
        x_chunk = coords[start:end]
        s_chunk, g_chunk, h_chunk = _compute_hessian_chunk(model, x_chunk)
        scalar_fields.append(s_chunk)
        gradients.append(g_chunk)
        hessians.append(h_chunk)

    scalar_field = torch.cat(scalar_fields, dim=0)
    gradient_field = torch.cat(gradients, dim=0)
    hessian_matrix = torch.cat(hessians, dim=0)
    return scalar_field, gradient_field, hessian_matrix


def _compute_hessian_chunk(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Hessian for a single chunk (training path).

    Uses ``retain_graph=True`` so outer losses can backpropagate later in the
    same iteration.

    Parameters
    ----------
    model : nn.Module
        Model mapping coordinates -> scalar potential.
    x : torch.Tensor
        Chunk of coordinates of shape ``(B, D)``.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        - Scalar field ``(B, 1)``
        - Gradient field ``(B, D)``
        - Hessian matrix ``(B, D, D)``
    """
    x = x.requires_grad_(True)
    scalar_field = model(x)
    grad_vector_field = torch.autograd.grad(
        outputs=scalar_field,
        inputs=x,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
    )[0]

    hessian_rows: List[torch.Tensor] = []
    for i in range(x.shape[1]):
        grad_i = grad_vector_field[:, i]
        hessian_row = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True,
            retain_graph=True,
        )[0]
        hessian_rows.append(hessian_row.unsqueeze(-1))

    hessian_matrix = torch.cat(hessian_rows, dim=-1)
    return scalar_field, grad_vector_field, hessian_matrix


def compute_hessian_eval(
    model: nn.Module,
    coords: torch.Tensor,
    chunk_size: int = 1024,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluation-only Hessian with careful graph handling and memory usage.

    Strategy
    --------
    - ``torch.no_grad()`` around the loop to avoid tracking param grads.
    - Re-enable grads for *inputs only* to obtain second derivatives.
    - Immediately detach and move results to CPU after each chunk.

    Parameters
    ----------
    model : nn.Module
        Model mapping coordinates -> scalar potential.
    coords : torch.Tensor
        Input coordinates of shape ``(N, D)``.
    chunk_size : int, default=1024
        Chunk size for evaluation.
    device : str | None, default=None
        Explicit device. Defaults to ``model.device`` if available.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        - Scalar field ``(N, 1)``
        - Gradient field ``(N, D)``
        - Hessian matrix ``(N, D, D)``

    Notes
    -----
    This function avoids creating third-order graphs (``create_graph=False`` for
    the second derivative rows) and ensures intermediates are freed promptly.
    """
    target_device = device if device is not None else getattr(model, "device", coords.device)
    model.to(target_device).eval()

    scalar_fields: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    hessians: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk_size):
            end = start + chunk_size
            x_chunk = coords[start:end].detach().clone().to(target_device).requires_grad_(True)

            # Enable grads for inputs to obtain second derivatives
            with torch.enable_grad():
                s_chunk, g_chunk, h_chunk = _compute_hessian_chunk_eval(model, x_chunk)

            scalar_fields.append(s_chunk.cpu().detach())
            gradients.append(g_chunk.cpu().detach())
            hessians.append(h_chunk.cpu().detach())

            # Cleanup device memory aggressively if on CUDA
            if torch.cuda.is_available():
                del s_chunk, g_chunk, h_chunk, x_chunk
                torch.cuda.empty_cache()

    scalar_field = torch.cat(scalar_fields, dim=0)
    gradient_field = torch.cat(gradients, dim=0)
    hessian_matrix = torch.cat(hessians, dim=0)
    return scalar_field, gradient_field, hessian_matrix


def _compute_hessian_chunk_eval(
    model: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Chunk-wise Hessian for evaluation path.

    Differs from the training path by not constructing third-order graphs
    (``create_graph=False``) for the second derivative rows.

    Parameters
    ----------
    model : nn.Module
        Model mapping coordinates -> scalar potential.
    x : torch.Tensor
        Chunk of coordinates of shape ``(B, D)``.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        - Scalar field ``(B, 1)``
        - Gradient field ``(B, D)``
        - Hessian matrix ``(B, D, D)``
    """
    scalar_field = model(x)
    grad_vector_field = torch.autograd.grad(
        outputs=scalar_field,
        inputs=x,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,   # need graph for second derivative
        retain_graph=True,
    )[0]

    hessian_rows: List[torch.Tensor] = []
    for i in range(x.shape[1]):
        grad_i = grad_vector_field[:, i]
        hessian_row = torch.autograd.grad(
            outputs=grad_i,
            inputs=x,
            grad_outputs=torch.ones_like(grad_i),
            retain_graph=True,
            create_graph=False,  # do not create 3rd-order graphs in eval
        )[0]
        hessian_rows.append(hessian_row.unsqueeze(-1))

        # Help GC on CUDA
        if torch.cuda.is_available():
            del hessian_row

    hessian_matrix = torch.cat(hessian_rows, dim=-1)
    return scalar_field, grad_vector_field, hessian_matrix
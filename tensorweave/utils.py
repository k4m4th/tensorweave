"""
Utility functions for geophysical field modeling and visualization.

This module collects common helpers you use with Neural Fourier Fields / FTG:
sampling, wavenumbers, noise injection, line-based subsampling, and plotting.

Highlights
----------
- Poisson-disk point selection (fast grid-hashing implementation).
- FFT-friendly 2D wavenumber grids.
- FTG noise injection by target SNR (per component).
- Greedy thinning (min-distance filter) and radius-based averaging.
- Clean plotting of selected Hessian components.

All functions are type-hinted and documented for Sphinx.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # kept for parity with your environment
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree


# --------------------------------------------------------------------------- #
#                               Image utilities                               #
# --------------------------------------------------------------------------- #

def hist_equalize(
    image: np.ndarray,
    nbins: int = 256,
    preserve_range: bool = False,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Histogram-equalize a NumPy image (optionally using a mask).

    The CDF is computed over masked (or full) intensities and then used to
    remap the whole image via linear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input image of any shape; NaNs are ignored when computing min/max.
    nbins : int, default=256
        Number of histogram bins.
    preserve_range : bool, default=False
        If True, rescale the equalized output back to the original [min, max].
    mask : np.ndarray | None, default=None
        Boolean mask specifying which pixels participate in the histogram/CDF.

    Returns
    -------
    np.ndarray
        Equalized image, same shape as ``image``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have the same shape as image.")

    sample = image[mask] if mask is not None else image.ravel()
    if sample.size == 0:
        return image.copy()

    vmin = np.nanmin(sample)
    vmax = np.nanmax(sample)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return image.copy()

    hist, bin_edges = np.histogram(sample, bins=nbins, range=(vmin, vmax), density=True)
    cdf = np.cumsum(hist)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mapped = np.interp(image.ravel(), bin_centers, cdf).reshape(image.shape)

    if preserve_range:
        mapped = mapped * (vmax - vmin) + vmin
    return mapped


# --------------------------------------------------------------------------- #
#                          Spectral / wavenumber grids                        #
# --------------------------------------------------------------------------- #

def make_wavenumber_grids_2d(
    spacing: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D angular-wavenumber grids (kx, ky) and radial magnitude (kr).

    Uses FFT-friendly frequencies via :func:`numpy.fft.fftfreq`. Angular
    wavenumbers are :math:`k = 2\\pi f`, where ``f`` are cycles per unit.

    Parameters
    ----------
    spacing : float
        Grid spacing (in same units as ``x_coords``/``y_coords``).
    x_coords : np.ndarray
        1D vector defining the x-grid (only the length is used).
    y_coords : np.ndarray
        1D vector defining the y-grid (only the length is used).

    Returns
    -------
    kx, ky, kr : (np.ndarray, np.ndarray, np.ndarray)
        2D arrays of shape ``(ny, nx)`` suitable for broadcasting with images.
    """
    nx = int(x_coords.shape[0])
    ny = int(y_coords.shape[0])
    if nx < 2 or ny < 2:
        raise ValueError("x_coords and y_coords must have length >= 2.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")

    fx = np.fft.fftfreq(nx, d=spacing)  # cycles / unit
    fy = np.fft.fftfreq(ny, d=spacing)
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    kx_grid, ky_grid = np.meshgrid(kx, ky)  # shape (ny, nx)
    kr = np.hypot(kx_grid, ky_grid)
    return kx_grid, ky_grid, kr


# --------------------------------------------------------------------------- #
#                           Sampling / point selection                        #
# --------------------------------------------------------------------------- #

def poisson_disk_indices(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    max_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Fast Poisson-disk sampling on an arbitrary set of (x,y) points.

    A uniform grid-hashing approach ensures that any two chosen points are
    at least ``radius`` apart by checking only the 8 neighboring cells.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of the same length N (flattened coordinates).
    radius : float
        Minimum separation between sampled points.
    max_points : int
        Maximum number of points to return (may return fewer if not feasible).
    seed : int | None, default=None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Indices (into ``x``/``y``) of selected points, dtype=int64.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if max_points <= 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    coords = np.c_[x, y]
    n = coords.shape[0]

    cell = radius / np.sqrt(2.0)  # r/√2 -> 8-neighborhood suffices
    xmin, ymin = coords.min(axis=0)
    ix = np.floor((coords[:, 0] - xmin) / cell).astype(np.int32)
    iy = np.floor((coords[:, 1] - ymin) / cell).astype(np.int32)

    order = rng.permutation(n)
    grid: dict[int, dict[int, int]] = {}
    chosen: List[int] = []
    r2 = radius * radius

    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),( 0,0),( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]

    for p in order:
        if len(chosen) >= max_points:
            break
        cx, cy = int(ix[p]), int(iy[p])
        ok = True
        for dx, dy in neighbors:
            gx, gy = cx + dx, cy + dy
            col = grid.get(gx)
            if col is None:
                continue
            q = col.get(gy)
            if q is None:
                continue
            dxv = coords[p, 0] - coords[q, 0]
            dyv = coords[p, 1] - coords[q, 1]
            if dxv * dxv + dyv * dyv < r2:
                ok = False
                break
        if ok:
            grid.setdefault(cx, {})[cy] = p
            chosen.append(p)

    return np.asarray(chosen, dtype=np.int64)


def poisson_disk_indices_naive(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    k: int,
) -> np.ndarray:
    """
    Naive (slow) Poisson-disk selection using cKDTree rebuilt each acceptance.

    This exists for parity with older scripts. Prefer :func:`poisson_disk_indices`.
    """
    coords = np.stack((x, y), axis=1)
    n = len(coords)
    sampled: List[int] = []
    accepted: List[np.ndarray] = []

    order = list(np.random.permutation(n))
    while order and len(sampled) < k:
        idx = order.pop()
        pt = coords[idx]
        if not accepted:
            sampled.append(idx)
            accepted.append(pt)
            continue
        tree = cKDTree(np.array(accepted))
        dist, _ = tree.query(pt[None, :], k=1)
        if float(dist[0]) >= radius:
            sampled.append(idx)
            accepted.append(pt)
    return np.asarray(sampled, dtype=np.int64)


# --------------------------------------------------------------------------- #
#                               Data utilities                                #
# --------------------------------------------------------------------------- #

def add_ftg_noise_by_snr(
    ftg: np.ndarray,
    snr_db: np.ndarray | float,
) -> np.ndarray:
    """
    Add Gaussian noise to FTG components to match a target SNR (dB).

    Parameters
    ----------
    ftg : np.ndarray
        Array of shape ``(N, 6)`` with the 6 tensor components.
    snr_db : float | array-like of shape (6,)
        Desired SNR per component in dB. A scalar is broadcast to all 6.

    Returns
    -------
    np.ndarray
        Noisy FTG array of shape ``(N, 6)``.
    """
    ftg = np.asarray(ftg)
    if ftg.ndim != 2 or ftg.shape[1] != 6:
        raise ValueError("ftg must have shape (N, 6).")

    snr = np.asarray(snr_db, dtype=float)
    if snr.size == 1:
        snr = np.full(6, float(snr))
    if snr.size != 6:
        raise ValueError("snr_db must be a scalar or an array-like of length 6.")

    power = np.mean(ftg**2, axis=0)          # per-component power
    snr_lin = 10.0 ** (snr / 10.0)
    noise_power = power / snr_lin
    std = np.sqrt(noise_power)

    noise = np.random.randn(*ftg.shape) * std
    return ftg + noise


def filter_by_min_distance(
    points: np.ndarray,
    data: np.ndarray,
    radius: float,
    randomize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy thinning: keep points such that no pair is closer than ``radius``.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, D)`` (D≥2). Only spatial distances are used.
    data : np.ndarray
        Associated data aligned to ``points`` along axis 0; returned subset matches.
    radius : float
        Minimum separation distance.
    randomize : bool, default=True
        If True, process points in random order; else in given order.

    Returns
    -------
    (np.ndarray, np.ndarray)
        - Kept points of shape ``(K, D)``.
        - Kept data of shape ``(K, ...)``.
    """
    if radius <= 0:
        return points.copy(), data.copy()

    tree = KDTree(points)
    n = len(points)
    removed = np.zeros(n, dtype=bool)
    order = np.arange(n)
    if randomize:
        np.random.shuffle(order)

    kept_idx: List[int] = []
    for idx in order:
        if removed[idx]:
            continue
        kept_idx.append(idx)
        neighbors = tree.query_ball_point(points[idx], radius)
        removed[neighbors] = True

    kept_idx = np.asarray(kept_idx, dtype=np.int64)
    return points[kept_idx], data[kept_idx]


def average_within_radius(
    points: np.ndarray,
    data: np.ndarray,
    radius: float,
    randomize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group points within ``radius`` and replace each group by its mean (points and data).

    This is useful for subsampling dense flight-line points while preserving
    the average signal within local neighborhoods.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, D)``.
    data : np.ndarray
        Array of shape ``(N, ...)`` aligned with points.
    radius : float
        Grouping radius.
    randomize : bool, default=True
        Randomize the processing order (affects grouping outcome slightly).

    Returns
    -------
    (np.ndarray, np.ndarray)
        - Averaged points, shape ``(G, D)``.
        - Averaged data, shape ``(G, ...)``.
    """
    tree = KDTree(points)
    n = len(points)
    processed = np.zeros(n, dtype=bool)

    avg_pts: List[np.ndarray] = []
    avg_dat: List[np.ndarray] = []

    order = np.arange(n)
    if randomize:
        np.random.shuffle(order)

    for idx in order:
        if processed[idx]:
            continue
        neighbors = tree.query_ball_point(points[idx], radius)
        valid = [i for i in neighbors if not processed[i]]
        grp_pts = points[valid]
        grp_dat = data[valid]
        avg_pts.append(np.mean(grp_pts, axis=0))
        avg_dat.append(np.mean(grp_dat, axis=0))
        processed[valid] = True

    return np.asarray(avg_pts), np.asarray(avg_dat)


def read_and_subsample_lines(
    path: str,
    column_idx: Sequence[int],
    radius: float = 5.0,
    delimiter: str = " ",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Read a whitespace-separated file and subsample points along flight lines.

    The file is assumed to be columns-only (no header). You specify which
    columns correspond to:
    - line id,
    - x, y, z (three columns),
    - and the remaining FTG (and/or other) columns.

    Points along each line are grouped within ``radius`` using
    :func:`average_within_radius`.

    Parameters
    ----------
    path : str
        Path to the file (e.g., CSV/space-delimited).
    column_idx : Sequence[int]
        Indices such that:
        - ``column_idx[0]`` = line_id column,
        - ``column_idx[1:4]`` = x, y, z columns,
        - the rest of the file is treated as data and concatenated.
    radius : float, default=5.0
        Averaging radius along each line.
    delimiter : str, default=" "
        Field delimiter.
    dtype : np.dtype, default=np.float32
        Data type for reading.

    Returns
    -------
    np.ndarray
        Array of shape ``(K, 1+3+M)`` with columns:
        ``[line_id, x, y, z, <data...>]``, after subsampling.
    """
    df = pd.read_csv(path, header=None, delimiter=delimiter, dtype=dtype)
    arr = df.to_numpy()

    line_id_col = int(column_idx[0])
    x_col, y_col, z_col = map(int, column_idx[1:4])

    line_ids = arr[:, line_id_col]
    # Normalize IDs to start at 0 but keep relative identity
    line_ids_norm = line_ids - np.nanmin(line_ids)

    xyz = arr[:, [x_col, y_col, z_col]]
    data = np.delete(arr, [line_id_col, x_col, y_col, z_col], axis=1)

    # group by line id
    grouped: dict[float, List[np.ndarray]] = {}
    for i, lid in enumerate(line_ids_norm):
        grouped.setdefault(lid, []).append(np.hstack([xyz[i], data[i]]))

    subsampled_rows: List[np.ndarray] = []
    for lid, rows in grouped.items():
        block = np.vstack(rows)                        # (Ni, 3+M)
        xyz_i, dat_i = block[:, :3], block[:, 3:]
        ss_xyz, ss_dat = average_within_radius(xyz_i, dat_i, radius)
        block_out = np.c_[np.full(ss_xyz.shape[0], lid), ss_xyz, ss_dat]
        subsampled_rows.append(block_out)

    out = np.vstack(subsampled_rows)
    return out


# --------------------------------------------------------------------------- #
#                                   Plotting                                   #
# --------------------------------------------------------------------------- #

_HESSIAN_INDEX = {
    "xx": 0,  "xy": 1,  "xz": 2,
    "yx": 3,  "yy": 4,  "yz": 5,
    "zx": 6,  "zy": 7,  "zz": 8,
}

def plot_hessian_components(
    predicted_hessian: torch.Tensor,
    active_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    components: Sequence[str] = ("xx", "xy", "xz"),
    cmap: str = "Spectral",
    equalize: bool = True,
    figsize: Tuple[float, float] = (12, 6),
) -> None:
    """
    Contour-plot selected Hessian components on a rectilinear grid.

    Parameters
    ----------
    predicted_hessian : torch.Tensor
        Tensor of shape ``(N, 3, 3)`` (or ``(N, 9)`` reshape-able) on CPU or CUDA.
    active_mask : np.ndarray
        Boolean or integer mask of length ``N`` indicating which *grid* points
        were evaluated (used to place values back onto the full grid).
    x_coords, y_coords : np.ndarray
        1D coordinate vectors defining the grid geometry for reshaping plot arrays.
        Final image will be ``(len(y_coords), len(x_coords))``.
    components : sequence of {"xx","xy","xz","yy","yz","zz"}, default=("xx","xy","xz")
        Which Hessian components to show (order matters).
    cmap : str, default="Spectral"
        Matplotlib colormap.
    equalize : bool, default=True
        If True, run histogram equalization on each component map for contrast.
    figsize : tuple(float, float), default=(12, 6)
        Figure size.

    Returns
    -------
    None
    """
    if predicted_hessian.dim() == 3 and predicted_hessian.shape[1:] == (3, 3):
        flat = predicted_hessian.reshape(-1, 9)
    elif predicted_hessian.dim() == 2 and predicted_hessian.shape[1] == 9:
        flat = predicted_hessian
    else:
        raise ValueError("predicted_hessian must be (N,3,3) or (N,9).")

    n = active_mask.shape[0]
    if flat.shape[0] != np.count_nonzero(active_mask):
        # We assume values correspond to True/active cells only.
        # If you passed the full-grid Hessian, set active_mask=np.ones(N, bool).
        pass

    # Reconstruct full-grid component array (N, 9) with NaNs elsewhere
    comp_full = np.full((n, 9), np.nan, dtype=np.float32)
    comp_full[active_mask] = flat.detach().cpu().numpy()

    nx = int(x_coords.shape[0])
    ny = int(y_coords.shape[0])

    # Prepare figure
    ncols = len(components)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, layout="compressed", sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, name in zip(axes, components):
        name = name.lower()
        if name not in _HESSIAN_INDEX:
            raise ValueError(f"Unknown component '{name}'.")
        idx = _HESSIAN_INDEX[name]

        img = comp_full[:, idx].reshape(ny, nx)
        if equalize:
            img = hist_equalize(img)

        cf = ax.contourf(x_coords, y_coords, img, levels=np.linspace(0, 1, 100), extend="both", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(f"H_{{{name}}}", pad=10)

    plt.show()


# --------------------------------------------------------------------------- #
#                                Module exports                                #
# --------------------------------------------------------------------------- #

__all__ = [
    "hist_equalize",
    "make_wavenumber_grids_2d",
    "poisson_disk_indices",
    "poisson_disk_indices_naive",
    "add_ftg_noise_by_snr",
    "filter_by_min_distance",
    "average_within_radius",
    "read_and_subsample_lines",
    "plot_hessian_components",
]

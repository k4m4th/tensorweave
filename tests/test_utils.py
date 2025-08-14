import numpy as np
import pandas as pd
import tensorweave as tw


def test_hist_equalize_basic():
    img = np.linspace(0, 1, 100).reshape(10, 10)
    eq = tw.hist_equalize(img, nbins=32, preserve_range=False)
    assert eq.shape == img.shape
    # still bounded in [0,1] with default settings
    assert float(eq.min()) >= 0.0 - 1e-6
    assert float(eq.max()) <= 1.0 + 1e-6

    # mask excludes half the pixels – should still return same shape
    mask = np.zeros_like(img, dtype=bool)
    mask[:5, :] = True
    eq2 = tw.hist_equalize(img, nbins=32, preserve_range=False, mask=mask)
    assert eq2.shape == img.shape


def test_make_wavenumber_grids_2d_shape_and_values():
    # simple 4x4 grid with spacing 1.0
    x = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    kx, ky, kr = tw.make_wavenumber_grids_2d(1.0, x, y)
    assert kx.shape == (len(y), len(x))
    assert ky.shape == (len(y), len(x))
    assert kr.shape == (len(y), len(x))

    # FFT frequency first column (ky when kx=0) should include 0
    assert np.isclose(ky[0, 0], 0.0)


def test_add_ftg_noise_by_snr_target_approx():
    rng = np.random.default_rng(0)
    ftg = rng.normal(size=(2000, 6)).astype(np.float64)
    target_snr_db = np.array([20, 25, 30, 20, 25, 30], dtype=float)
    noisy = tw.add_ftg_noise_by_snr(ftg, target_snr_db)

    # estimate SNR of noisy signal: signal power / noise power
    # We don't have clean "signal" here, so compute noise by difference
    noise = noisy - ftg
    sig_pow = (ftg ** 2).mean(axis=0)
    noi_pow = (noise ** 2).mean(axis=0)
    snr_db_est = 10.0 * np.log10(sig_pow / (noi_pow + 1e-12))

    # Allow a few dB tolerance due to finite sampling
    assert np.all(np.abs(snr_db_est - target_snr_db) < 2.5)


def test_filter_and_average_within_radius():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(200, 3))
    dat = rng.normal(size=(200, 6))
    r = 5.0

    kept_pts, kept_dat = tw.filter_by_min_distance(pts, dat, radius=r, randomize=True)
    assert kept_pts.shape[0] == kept_dat.shape[0]
    # check min distance
    if kept_pts.shape[0] > 1:
        diff = kept_pts[None, :, :] - kept_pts[:, None, :]
        d2 = np.sum(diff ** 2, axis=-1) + np.eye(kept_pts.shape[0]) * 1e9
        assert float(np.min(d2)) >= (r - 1e-6) ** 2

    avg_pts, avg_dat = tw.average_within_radius(pts, dat, radius=r, randomize=True)
    assert avg_pts.shape[0] == avg_dat.shape[0]
    # should not increase the number of points
    assert avg_pts.shape[0] <= pts.shape[0]


def test_read_and_subsample_lines(tmp_path):
    # build a fake dataset with line IDs and (x,y,z) + 6 FTG columns
    rng = np.random.default_rng(0)
    n = 200
    line_ids = np.repeat(np.arange(4), n // 4)
    xyz = rng.uniform(0, 1000, size=(n, 3))
    ftg = rng.normal(size=(n, 6))
    arr = np.c_[line_ids, xyz, ftg].astype(np.float32)

    p = tmp_path / "synthetic.txt"
    # space-delimited, no header
    import pandas as pd
    pd.DataFrame(arr).to_csv(p, index=False, header=False, sep=" ")

    out = tw.read_and_subsample_lines(str(p), column_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], radius=10.0)
    # line_id + 3 coords + 6 ftg = 10 columns
    assert out.shape[1] == 10
    # fewer or equal samples than original
    assert out.shape[0] <= n
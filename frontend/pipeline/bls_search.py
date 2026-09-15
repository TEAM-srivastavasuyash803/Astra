"""
Box Least Squares (BLS) Coarse-to-Fine Transit Search Module
Complies with robust MAD-based transit search specifications.
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares


PERIOD_MIN = 3.0
PERIOD_MAX = 400.0
DEFAULT_N_COARSE = 20000
DEFAULT_N_PEAKS = 8
DEFAULT_N_FINE = 600
DEFAULT_DURATIONS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])  # days


def compute_sde(power, peak_idx):
    """
    Signal Detection Efficiency (SDE):
    Peak height above periodogram floor in robust (MAD-based) standard deviations.
    MAD is used so a strong transit peak does not inflate its own noise estimate.
    """
    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med))
    if mad > 0 and np.isfinite(mad):
        return float((power[peak_idx] - med) / (1.4826 * mad))
    return 0.0


def search_transit_bls(
    t,
    f,
    n_coarse=DEFAULT_N_COARSE,
    n_peaks=DEFAULT_N_PEAKS,
    n_fine=DEFAULT_N_FINE,
    durations=DEFAULT_DURATIONS,
    verbose=False,
):
    """
    Executes a coarse-to-fine Box Least Squares search over light curve (t, f).
    Returns best candidate parameters, coarse periodogram samples, and candidate peaks.
    """
    baseline = float(t.max() - t.min())
    pmax = min(PERIOD_MAX, baseline / 3.0)  # enforce >= 3 observed transits

    if pmax <= PERIOD_MIN:
        return {
            "period": np.nan,
            "depth_ppm": np.nan,
            "duration_hours": np.nan,
            "t0": np.nan,
            "sde": 0.0,
            "coarse_periods": [],
            "coarse_powers": [],
            "top_peaks": [],
        }

    bls = BoxLeastSquares(t, f)

    # 1. Coarse sweep (log-spaced)
    coarse_periods = np.exp(np.linspace(np.log(PERIOD_MIN), np.log(pmax), n_coarse))
    coarse_res = bls.power(coarse_periods, durations, objective="likelihood")
    coarse_power = np.asarray(coarse_res.power)

    # 2. Pick top well-separated peaks (each excludes +/- 10% around itself)
    sort_idx = np.argsort(coarse_power)[::-1]
    peaks = []
    used = np.zeros(len(coarse_periods), dtype=bool)

    for i in sort_idx:
        if used[i]:
            continue
        peaks.append(i)
        p_val = coarse_periods[i]
        lo = np.searchsorted(coarse_periods, p_val * 0.90)
        hi = np.searchsorted(coarse_periods, p_val * 1.10)
        used[lo:hi] = True
        if len(peaks) >= n_peaks:
            break

    # 3. Fine sweep around each candidate coarse peak (+/- 2%)
    best = None
    top_candidates = []

    for coarse_idx in peaks:
        p0 = coarse_periods[coarse_idx]
        width = 0.02 * p0
        fine_periods = np.linspace(p0 - width, p0 + width, n_fine)
        fine_periods = fine_periods[fine_periods > PERIOD_MIN]

        if len(fine_periods) < 10:
            continue

        fine_res = bls.power(fine_periods, durations, objective="likelihood")
        fine_power = np.asarray(fine_res.power)
        j = int(np.nanargmax(fine_power))

        candidate_sde = compute_sde(coarse_power, coarse_idx)

        cand_info = {
            "period": float(fine_res.period[j]),
            "depth_ppm": float(fine_res.depth[j] * 1e6),
            "duration_hours": float(fine_res.duration[j] * 24),
            "t0": float(fine_res.transit_time[j]),
            "sde": candidate_sde,
            "coarse_period": float(p0),
        }
        top_candidates.append(cand_info)

        if best is None or candidate_sde > best["sde"]:
            best = cand_info
            best["fine_periods"] = fine_periods
            best["fine_power"] = fine_power

        if verbose:
            print(f"  Peak P={p0:8.3f}d -> Refined {fine_res.period[j]:8.4f}d, SDE={candidate_sde:5.1f}")

    if best is None:
        best = {
            "period": np.nan,
            "depth_ppm": np.nan,
            "duration_hours": np.nan,
            "t0": np.nan,
            "sde": 0.0,
        }

    # Subsample coarse periodogram for efficient frontend rendering (e.g. 500 points)
    step = max(1, len(coarse_periods) // 600)
    best["coarse_periods_sample"] = coarse_periods[::step].tolist()
    best["coarse_powers_sample"] = coarse_power[::step].tolist()
    best["top_peaks"] = top_candidates

    return best

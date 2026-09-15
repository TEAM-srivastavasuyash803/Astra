"""
Kepler SAP Flux Ingestion & Quality Masking Module
"""
import os
import numpy as np
import pandas as pd


def load_kepler_data(file_path):
    """
    Load raw Kepler parquet file.
    Expected schema: time, flux, flux_err, quality, quarter
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_parquet(file_path)
    required = ["time", "flux", "quality", "quarter"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {file_path}")
    return df


def clean_light_curve(df, min_cadences=1000, strict_quality=True):
    """
    Ingestion & quality filtering:
    - Drops non-finite flux and bad quality flags.
    - Divides each quarter by its own median to remove inter-quarter jumps.
    - Discards star if fewer than min_cadences points survive.
    """
    if df is None or len(df) == 0:
        return None

    if strict_quality:
        mask = (df["quality"].values == 0) & np.isfinite(df["flux"].values)
    else:
        # Mask fatal flags (cosmic rays, attitude tweak, safe mode)
        fatal = 1 | 2 | 4 | 8 | 16 | 32 | 128 | 2048
        mask = ((df["quality"].values & fatal) == 0) & np.isfinite(df["flux"].values)

    t = df["time"].values[mask].astype(float)
    f = df["flux"].values[mask].astype(float)
    q = df["quarter"].values[mask].astype(int)

    if "flux_err" in df.columns:
        e = df["flux_err"].values[mask].astype(float)
    else:
        e = np.ones_like(f) * 1e-4

    if len(t) < min_cadences:
        return None

    # Sort time ascending
    order = np.argsort(t)
    t, f, e, q = t[order], f[order], e[order], q[order]

    # Quarter normalization
    norm_flux = np.copy(f)
    norm_err = np.copy(e)
    for q_val in np.unique(q):
        s = q == q_val
        med = np.nanmedian(f[s])
        if med > 0 and np.isfinite(med):
            norm_flux[s] = f[s] / med
            norm_err[s] = e[s] / med
        else:
            norm_flux[s] = 1.0

    return {
        "time": t,
        "flux": norm_flux,
        "flux_err": norm_err,
        "quarter": q,
        "raw_flux": f,
        "baseline_days": float(t.max() - t.min()),
        "cadence_count": len(t),
    }

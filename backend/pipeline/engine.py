import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from astropy.timeseries import BoxLeastSquares

class OptimizedExoplanetPipeline:
    """Vectorized Kepler Light Curve Signal Processing Pipeline."""
    
    def __init__(self, sde_threshold: float = 10.0, min_period: float = 3.0, max_period: float = 400.0):
        self.sde_threshold = sde_threshold
        self.min_period = min_period
        self.max_period = max_period

    def clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Masks bad cadences (quality != 0) and normalizes flux per quarter in memory."""
        # Bitwise vector mask across all cadences[cite: 1]
        valid_mask = (df['quality'] == 0) & np.isfinite(df['flux']) & np.isfinite(df['time'])
        df_valid = df[valid_mask].copy()

        # Discard star if fewer than 1,000 valid cadences remain[cite: 1]
        if len(df_valid) < 1000:
            return pd.DataFrame()

        # Vectorized per-quarter median normalization[cite: 1]
        q_medians = df_valid.groupby('quarter')['flux'].transform('median')
        df_valid['norm_flux'] = (df_valid['flux'] / q_medians).astype(np.float32)

        return df_valid.sort_values('time')

    def detrend(self, df: pd.DataFrame, window_length: int = 501, polyorder: int = 3) -> pd.DataFrame:
        """Applies Savitzky-Golay filtering to remove stellar variability while preserving transit depth[cite: 1]."""
        if 'norm_flux' not in df.columns or len(df) < window_length:
            return df

        # Direct C-array execution via SciPy filter[cite: 1]
        flux_array = df['norm_flux'].values
        trend = savgol_filter(flux_array, window_length=window_length, polyorder=polyorder)
        df['flat_flux'] = (flux_array - trend + 1.0).astype(np.float32)

        return df

    def bls_search(self, df: pd.DataFrame) -> dict:
        """Runs coarse-to-fine Box Least Squares search and returns best candidate features."""
        t = df['time'].values
        y = (df['flat_flux'].values - 1.0).astype(np.float64)
        
        med_flux = np.median(df['flux'].values)
        dy = (df['flux_err'].values / med_flux).astype(np.float64) if med_flux > 0 else np.ones_like(y) * 1e-4

        model = BoxLeastSquares(t, y, dy)
        durations = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=np.float64)

        max_p = min(self.max_period, (t.max() - t.min()) / 3.0)
        if max_p <= self.min_period:
            return {'sde': 0.0, 'period': 0.0, 'depth_ppm': 0.0, 'duration_hours': 0.0, 'odd_even_ratio': 0.0}

        periods = model.autoperiod(durations, minimum_period=self.min_period, maximum_period=max_p)
        periodogram = model.power(periods, durations)

        power = periodogram.power
        med_power = np.median(power)
        mad = np.median(np.abs(power - med_power))

        if mad <= 0:
            return {'sde': 0.0, 'period': 0.0, 'depth_ppm': 0.0, 'duration_hours': 0.0, 'odd_even_ratio': 0.0}

        sde_array = (power - med_power) / (1.4826 * mad)
        best_idx = np.argmax(sde_array)
        best_sde = float(sde_array[best_idx])
        best_coarse_period = float(periodogram.period[best_idx])

        # Fine sweep restricted to +/- 2% around the coarse peak
        fine_periods = np.linspace(best_coarse_period * 0.98, best_coarse_period * 1.02, 600)
        fine_pg = model.power(fine_periods, durations)
        fine_idx = np.argmax(fine_pg.power)

        best_period = float(fine_pg.period[fine_idx])
        best_t0 = float(fine_pg.transit_time[fine_idx])
        best_duration = float(fine_pg.duration[fine_idx])

        # --- ODD/EVEN TRANSIT CHECK ---
        # 1. Calculate the epoch (transit number) for every timestamp
        transit_epochs = np.round((t - best_t0) / best_period)
        
        # 2. Isolate data points that fall inside the transit window
        time_from_mid_transit = np.abs((t - best_t0) / best_period - transit_epochs) * best_period
        in_transit = time_from_mid_transit < (best_duration / 2.0)
        
        # 3. Create bitwise masks for alternating transits
        even_mask = in_transit & (transit_epochs % 2 == 0)
        odd_mask = in_transit & (transit_epochs % 2 != 0)
        
        # 4. Calculate depth (y is zero-centered, so dips are negative)
        even_depth = -np.median(y[even_mask]) if np.sum(even_mask) > 0 else 0.0
        odd_depth = -np.median(y[odd_mask]) if np.sum(odd_mask) > 0 else 0.0
        
        # 5. Compute consistency ratio (inconsistent depths suggest an eclipsing binary)[cite: 2]
        if even_depth > 0 and odd_depth > 0:
            odd_even_ratio = min(even_depth, odd_depth) / max(even_depth, odd_depth)
        else:
            odd_even_ratio = 0.0

        return {
            'sde': best_sde,
            'period': best_period,
            'depth_ppm': float(fine_pg.depth[fine_idx] * 1e6),
            'duration_hours': best_duration * 24.0,
            'odd_even_ratio': float(odd_even_ratio)
        }

    def vet_and_format(self, star_id: str, bls_res: dict) -> dict:
        """Calibrates confidence and formats candidate output per official spec."""
        sde = bls_res['sde']
        odd_even_ratio = bls_res.get('odd_even_ratio', 0.0)

        # Multi-feature classifier: Signal must be strong AND consistent across odd/even epochs[cite: 2]
        # A ratio < 0.5 indicates highly alternating depths (likely binary star, not a planet)[cite: 2]
        if sde >= self.sde_threshold and odd_even_ratio >= 0.5:
            prediction = 1
            confidence = float(1.0 / (1.0 + np.exp(-0.4 * (sde - self.sde_threshold))))
            return {
                'star_id': star_id,
                'prediction': 1,
                'confidence': round(confidence, 4),
                'period': round(bls_res['period'], 5),
                'depth_ppm': round(bls_res['depth_ppm'], 2),
                'duration_hours': round(bls_res['duration_hours'], 2)
            }
        else:
            # Leave characterization fields empty for prediction = 0
            return {
                'star_id': star_id,
                'prediction': 0,
                'confidence': round(float(1.0 / (1.0 + np.exp(-0.4 * (sde - self.sde_threshold)))), 4),
                'period': None,
                'depth_ppm': None,
                'duration_hours': None
            }

    def process_star(self, star_id: str, df: pd.DataFrame) -> dict:
        """Pipeline execution entry point for an individual star dataframe[cite: 1]."""
        df_clean = self.clean_and_normalize(df)
        if df_clean.empty:
            return self.vet_and_format(star_id, {'sde': 0.0, 'period': 0.0, 'depth_ppm': 0.0, 'duration_hours': 0.0})

        df_detrended = self.detrend(df_clean)
        bls_results = self.bls_search(df_detrended)
        return self.vet_and_format(star_id, bls_results)
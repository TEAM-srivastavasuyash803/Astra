"""
Planetary Characterisation & Astrophysical Parameter Estimation Module
Calculates physical and orbital derived parameters.
"""
import numpy as np


def characterize_candidate(
    period,
    depth_ppm,
    duration_hours,
    t0=0.0,
    teff=5778.0,
    logg=4.438,
    radius=1.0,
    kepmag=12.0,
):
    """
    Computes required submission parameters and derived astrophysical attributes.
    """
    if np.isnan(period) or period <= 0 or np.isnan(depth_ppm) or depth_ppm <= 0:
        return {
            "period": None,
            "depth_ppm": None,
            "duration_hours": None,
            "t0": None,
            "rp_rs": None,
            "planet_radius_earth": None,
            "semi_major_axis_au": None,
            "stellar_flux_earth": None,
            "teq_k": None,
            "habitable_zone": False,
            "planet_class": "Unknown",
        }

    # Clean default stellar parameters if missing or nan
    teff = float(teff) if (teff is not None and np.isfinite(teff) and teff > 1000) else 5778.0
    logg = float(logg) if (logg is not None and np.isfinite(logg) and logg > 1.0) else 4.438
    radius = float(radius) if (radius is not None and np.isfinite(radius) and radius > 0.1) else 1.0
    kepmag = float(kepmag) if (kepmag is not None and np.isfinite(kepmag)) else 12.0

    # 1. Radius ratio (Rp / Rs)
    depth_fraction = max(1e-7, depth_ppm * 1e-6)
    rp_rs = float(np.sqrt(depth_fraction))

    # 2. Planet radius in Earth radii (1 R_sun = 109.2 R_earth)
    r_earth = float(rp_rs * radius * 109.2)

    # 3. Stellar mass from logg and radius: M / M_sun = 10^(logg - 4.438) * (R / R_sun)^2
    m_star = float(10 ** (logg - 4.438) * (radius ** 2))
    m_star = max(0.2, min(5.0, m_star))

    # 4. Semi-major axis (AU) via Kepler's 3rd Law: a^3 = M_star * (P / 365.25)^2
    semi_major_axis_au = float((m_star * (period / 365.25) ** 2) ** (1.0 / 3.0))

    # 5. Insolation flux relative to Earth: S = (R_* / R_sun)^2 * (Teff / 5778)^4 / a^2
    s_earth = float((radius ** 2) * ((teff / 5778.0) ** 4) / max(1e-4, semi_major_axis_au ** 2))

    # 6. Planet equilibrium temperature (assuming Bond albedo = 0.3)
    # Teq = Teff * sqrt(R_star / (2 * a)) * (1 - A)^(1/4)
    # R_star in AU: 1 R_sun = 0.00465047 AU
    r_star_au = radius * 0.00465047
    teq_k = float(teff * np.sqrt(r_star_au / (2.0 * max(1e-4, semi_major_axis_au))) * (0.7 ** 0.25))

    # 7. Habitable zone check (approximate Kopparapu runaway greenhouse to maximum greenhouse: 0.32 - 1.78 S_earth)
    habitable_zone = bool(0.32 <= s_earth <= 1.78)

    # 8. Classification
    if r_earth <= 1.25:
        if period >= 180.0 and habitable_zone:
            planet_class = "Earth Analog"
        else:
            planet_class = "Rocky Earth-Sized"
    elif r_earth <= 2.0:
        planet_class = "Super-Earth"
    elif r_earth <= 4.0:
        planet_class = "Sub-Neptune"
    elif r_earth <= 10.0:
        planet_class = "Neptunian Giant"
    else:
        planet_class = "Jovian Gas Giant"

    return {
        "period": round(float(period), 5),
        "depth_ppm": round(float(depth_ppm), 1),
        "duration_hours": round(float(duration_hours), 3),
        "t0": round(float(t0), 4),
        "rp_rs": round(rp_rs, 5),
        "planet_radius_earth": round(r_earth, 2),
        "semi_major_axis_au": round(semi_major_axis_au, 3),
        "stellar_flux_earth": round(s_earth, 2),
        "teq_k": round(teq_k, 0),
        "habitable_zone": habitable_zone,
        "planet_class": planet_class,
        "host_teff": round(teff, 0),
        "host_radius": round(radius, 2),
        "host_logg": round(logg, 3),
    }

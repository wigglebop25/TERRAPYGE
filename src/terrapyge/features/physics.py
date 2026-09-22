"""Physics-based feature computation for TERRAPYGE.

Implements the infinite-slope stability model and Newmark displacement
for earthquake-induced landslide susceptibility.

References:
- Newmark, N.M. (1965). Effects of earthquakes on dams and embankments.
- Jibson, R.W. (2007). Regression models for estimating coseismic
  landslide displacement. Engineering Geology, 91(2-4), 209-218.
"""

import numpy as np

# Physical constants
G = 9.81  # m/s^2 or kN per kN-mass


def derive_geotechnical_params(clay_pct, sand_pct, bulk_density_raw):
    """Derive geotechnical parameters from soil texture and bulk density.

    Note: empirical (pedotransfer-style) approximations for weathered
    tropical residual soils in the absence of site-specific geotechnical
    testing. Cohesion is kept low (shallow colluvial failure planes) and
    friction angle follows sand content. Must be recalibrated wherever
    field or laboratory data exist.

    Args:
        clay_pct: Clay content (%).
        sand_pct: Sand content (%).
        bulk_density_raw: Bulk density (cg/cm^3 as exported by OpenLandMap).

    Returns:
        gamma_kNm3: Soil unit weight (kN/m^3), clamped to [14, 20].
        c_prime_kpa: Effective cohesion (kPa).
        phi_prime_deg: Effective friction angle (degrees), clamped [25, 38].
    """
    bd = np.asarray(bulk_density_raw, dtype=float)
    bd_gcm3 = bd / 100.0  # assume OpenLandMap values are cg/cm^3
    gamma = bd_gcm3 * G
    gamma = np.clip(gamma, 14.0, 20.0)
    gamma = np.where(np.isfinite(gamma), gamma, 17.0)

    # Low effective cohesion typical of shallow colluvial failure planes
    c_prime = 2.0 + 0.1 * np.clip(clay_pct, 0, 100)
    c_prime = np.clip(c_prime, 2.0, 10.0)

    # Friction angle governed mainly by sand content
    phi_prime = 28.0 + 0.1 * np.clip(sand_pct, 0, 100)
    phi_prime = np.clip(phi_prime, 25.0, 38.0)

    return gamma, c_prime, phi_prime


def static_fs(slope_deg, c_prime_kpa, phi_prime_deg, gamma_kNm3, z_m,
              ru=0.15, fs_max=1000.0):
    """Static Factor of Safety for infinite slope with pore-pressure ratio.

    FS = [c' + (gamma*z*cos^2(a) - u) * tan(phi')] / (gamma*z*sin(a)*cos(a))
    u  = ru * gamma * z    (pore-pressure ratio ru)

    Units: c' in kPa (= kN/m^2); gamma in kN/m^3; z in m.

    FS is clipped to [1.0, fs_max]:
    - lower bound 1.0: existing slopes are assumed to be at limit equilibrium;
      any excess driving stress has already been relieved by prior failures.
    - upper bound for numerical stability on very gentle terrain.
    """
    a = np.radians(np.clip(slope_deg, 0.1, 89.0))
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    tan_phi = np.tan(np.radians(phi_prime_deg))
    u = ru * gamma_kNm3 * z_m
    numerator = c_prime_kpa + (gamma_kNm3 * z_m * cos_a ** 2 - u) * tan_phi
    denominator = gamma_kNm3 * z_m * sin_a * cos_a
    fs = numerator / np.maximum(denominator, 1e-6)
    return np.clip(fs, 1.0, fs_max)


def critical_acceleration_g(fs, slope_deg, ac_floor=0.01):
    """Critical (yield) acceleration in units of g.

    Ac = (FS - 1) * sin(alpha), floored at ac_floor to keep the
    Newmark regression finite for near-zero yield accelerations.
    """
    a = np.radians(np.clip(slope_deg, 0.1, 89.0))
    ac = (fs - 1.0) * np.sin(a)
    return np.clip(ac, ac_floor, None)


def arias_intensity(PGA_ms2, duration_s=10.0):
    """Approximate Arias intensity from PGA and significant duration.

    Triangle approximation of acceleration time history:
        Ia = pi * PGA^2 * D / (6 * g)

    Returns Ia in m/s.
    """
    return (np.pi * (PGA_ms2 ** 2) * duration_s) / (6.0 * G)


def newmark_displacement_cm(arias_ms, ac_g):
    """Jibson (2007) Model 1 regression for Newmark displacement.

    log10(Dn) = 0.561*log10(Ia) - 1.120*log10(Ac) - 0.552
    Ia in m/s, Ac in g, Dn in cm.

    Returns 0 where Ac <= 0 (no yield acceleration => no downslope movement)
    or where inputs are non-positive.
    """
    ia = np.asarray(arias_ms, dtype=float)
    ac = np.asarray(ac_g, dtype=float)
    ia = np.broadcast_to(ia, ac.shape)
    dn = np.zeros_like(ac, dtype=float)
    valid = (ia > 0) & (ac > 1e-6)
    log_dn = (0.561 * np.log10(ia[valid])
              - 1.120 * np.log10(ac[valid])
              - 0.552)
    dn[valid] = 10.0 ** log_dn
    return dn


def physics_labels(dn_cm, thresholds=(2.0, 5.0, 15.0)):
    """Map Newmark displacement to susceptibility class.

    thresholds = (t_moderate, t_high, t_very_high) in cm.
    Default: < 2 Low; 2-5 Moderate; 5-15 High; >= 15 Very High.
    """
    t1, t2, t3 = thresholds
    dn = np.asarray(dn_cm, dtype=float)
    labels = np.zeros_like(dn, dtype=int)
    labels[(dn >= t1) & (dn < t2)] = 1
    labels[(dn >= t2) & (dn < t3)] = 2
    labels[dn >= t3] = 3
    return labels

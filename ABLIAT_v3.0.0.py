from __future__ import annotations
import math
import numpy as np
import contextlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---- CONSTANTS ----
MU_EARTH = 3.986004418e14
R_EARTH  = 6378.137e3
STEFAN_BOLTZMANN = 5.670374e-8


# ---- HELPER FUNCTIONS ----
def dv_to_perigee_300km(h_circ_km: float, hp_km: float = 300.0) -> float:
    """
    Calculates the delta-v required to lower the perigee of a circular orbit.

    This function computes the single, tangential, impulsive delta-v (in m/s)
    needed to transition from an initial circular orbit at h_circ_km to an
    elliptical transfer orbit. The apogee of this new orbit remains at the
    original altitude, while the perigee is lowered to hp_km.

    It uses the vis-viva equation to find the velocity at the apogee of the
    new transfer orbit and subtracts it from the original circular orbit
    velocity.

    Args:
        h_circ_km (float): The initial circular orbit altitude in kilometers.
        hp_km (float, optional): The target perigee altitude in kilometers.
                                 Defaults to 300.0.

    Returns:
        float: The required delta-v in m/s.
    """
    r0 = R_EARTH + h_circ_km * 1e3
    rp = R_EARTH + hp_km * 1e3
    a  = 0.5 * (r0 + rp)
    v_circ = (MU_EARTH / r0) ** 0.5
    v_ap   = (MU_EARTH * (2.0/r0 - 1.0/a)) ** 0.5
    return float(v_circ - v_ap)

def spot_diameter(L: float, D: float, lam: float, M2: float = 1.0, jitter_urad: float = 0.0,
                  model: str = "gaussian_1e2") -> float:
    """
    Calculates the effective on-target spot diameter from a laser.

    This function models the far-field beam divergence based on aperture
    diameter, wavelength, and beam quality (M2). It allows for different
    beam profile definitions (e.g., Gaussian 1/e^2, Airy disk) and
    accounts for pointing jitter by adding it in quadrature to the core
    beam divergence.

    Args:
        L (float): Slant range to the target in meters.
        D (float): Effective diameter of the transmitter aperture in meters.
        lam (float): Laser wavelength in meters.
        M2 (float, optional): Beam quality factor (1.0 = perfect diffraction limit).
                              Defaults to 1.0.
        jitter_urad (float, optional): Pointing jitter in microradians (µrad).
                                       Defaults to 0.0.
        model (str, optional): The beam profile model to use for the
                               divergence constant 'K'.
                               Defaults to "gaussian_1e2".

    Returns:
        float: The effective on-target spot diameter in meters.
    """
    K_map = {
        "gaussian_1e2":    4.0 / math.pi,          # 1/e^2 Gaussian
        "airy_first_null": 2.44,                   # Airy first null
        "gaussian_fwhm":   (4.0 / math.pi) * 0.5887,
        "airy_fwhm":       1.03,
    }
    K = K_map.get(model, 4.0 / math.pi)
    theta_core = (K / 2.0) * M2 * lam / max(D, 1e-9)    # rad
    theta_jit  = (jitter_urad or 0.0) * 1e-6            # rad
    theta_eff  = math.hypot(theta_core, theta_jit)
    return 2.0 * theta_eff * L                          # diameter [m]

def fluence_J_cm2(E_abs_J: float, spot_diameter_m: float) -> float:
    """
    Calculates the laser fluence in J/cm².

    This helper function converts the spot diameter from meters to centimeters
    and calculates the fluence (Energy / Area) for a circular spot, assuming
    a uniform "top-hat" energy distribution.

    Args:
        E_abs_J (float): The absorbed energy per pulse in Joules.
        spot_diameter_m (float): The on-target spot diameter in meters.

    Returns:
        float: The calculated fluence in Joules per square centimeter (J/cm²).
    """
    d_cm = spot_diameter_m * 100.0
    return 4.0 * E_abs_J / (math.pi * d_cm**2) if d_cm > 0 else float("inf")

def cm_model(F_cm2: float, tau_s: float, cmin: float = 1e-4, cmax: float = 1e-3,
             F0: float = 20.0, a: float = 0.6) -> float:
    """
    Calculates the momentum coupling coefficient (C_m) based on fluence.

    This function provides a simple, saturating model for C_m as a function of
    fluence (F_cm2), bounded by cmin and cmax. This serves as a placeholder
    for a more complex, data-driven model.

    Args:
        F_cm2 (float): The absorbed fluence in J/cm².
        tau_s (float): The pulse duration in seconds. (Note: tau_s is an
                       argument but not used in the current formula).
        cmin (float, optional): The minimum C_m value (at F=0). Defaults to 1e-4.
        cmax (float, optional): The saturation C_m value (at F=inf). Defaults to 1e-3.
        F0 (float, optional): The characteristic fluence (in J/cm²) where C_m
                              reaches half its max. Defaults to 20.0.
        a (float, optional): The exponent controlling the steepness of the
                             saturation. Defaults to 0.6.

    Returns:
        float: The calculated momentum coupling coefficient (C_m) in N·s/J.
    """
    if F_cm2 <= 0:
        return cmin
    Cm = cmin + (cmax - cmin) * (F_cm2**a) / (F0**a + F_cm2**a)
    return float(np.clip(Cm, cmin, cmax))

def design_metrics(x: dict, Cm_override: float | None = None) -> dict:
    """
    Calculates all key performance metrics for a given laser system design.

    This function takes a dictionary of design parameters (aperture, power,
    target info, etc.) and computes a full set of derived performance
    metrics, including spot size, fluence, number of pulses, total energy,
    power requirements, and mission time ("laser-on" time).

    It determines which C_m value to use based on the 'cm_mode' in the input
    dictionary or an optional override.

    Args:
        x (dict): A dictionary containing all input design parameters
                  (e.g., 'D', 'L', 'E_pulse', 'f_rep', 'm_target', etc.).
        Cm_override (float, optional): A specific C_m value to use, bypassing
                                       the 'cm_mode' logic. Defaults to None.

    Returns:
        dict: A dictionary of all calculated performance metrics
              (e.g., 'spot_diameter_m', 'fluence_J_cm2', 'N_pulses',
              'P_avg_W', 'mission_time_s', etc.).
    """
    D            = float(x['D'])
    L            = float(x['L'])
    lam          = float(x['lam'])
    E_pulse      = float(x['E_pulse'])
    tau          = float(x['tau'])
    f_rep        = float(x['f_rep'])
    eta_abs      = float(x['eta_abs'])
    M2           = float(x['M2'])
    jitter_urad  = float(x['jitter_urad'])
    m_target     = float(x['m_target'])
    dv_target    = float(x['dv_target'])
    cm_mode      = x.get('cm_mode', 'range')
    cm_fixed     = float(x.get('cm_fixed', 3e-4))
    cm_min       = float(x.get('cm_min', 1e-4))
    cm_max       = float(x.get('cm_max', 1e-3))
    eta_wall     = max(1e-3, float(x['eta_wallplug']))
    spot_model   = x.get('spot_model', 'gaussian_1e2')

    d_m   = spot_diameter(L, D, lam, M2, jitter_urad, model=spot_model)
    E_abs = E_pulse * eta_abs
    F     = fluence_J_cm2(E_abs, d_m)

    if Cm_override is not None:
        Cm = float(Cm_override)
    else:
        if cm_mode == 'model':
            Cm = cm_model(F, tau)
        elif cm_mode == 'fixed':
            Cm = cm_fixed
        elif cm_mode == 'range':
            Cm = cm_min  # conservative
        else:
            Cm = cm_fixed

    delta_p   = Cm * E_abs                    # N·s per pulse
    N_pulses  = (m_target * dv_target) / max(delta_p, 1e-20)
    E_total   = N_pulses * E_pulse            # J (optical)
    P_peak    = E_pulse / max(tau, 1e-20)     # W
    P_avg     = E_pulse * f_rep               # W (optical)
    P_elec    = P_avg / eta_wall              # W (electrical)
    T_mission = N_pulses / max(f_rep, 1e-12)  # s

    return {
        'spot_diameter_m': d_m,
        'fluence_J_cm2':   F,
        'Cm':              Cm,
        'delta_p_per_pulse': delta_p,
        'N_pulses':        N_pulses,
        'E_total_J':       E_total,
        'P_peak_W':        P_peak,
        'P_avg_W':         P_avg,
        'P_elec_avg_W':    P_elec,
        'mission_time_s':  T_mission,
    }

def cost_model(x: dict, c: dict, Cm_use: float | None = None) -> float:
    """
    Calculates the total mission cost based on system design and metrics.

    This model estimates the total program cost by summing Capital
    Expenditures (CAPEX) and Operational Expenditures (OPEX).

    CAPEX includes:
    - C_optics: Cost of the primary aperture, scaling with area ($/m²).
    - C_laser: Cost of the laser, scaling with average *electrical* power
               ($/kW) and subject to a cost floor.
    - C_aperture: A non-linear cost for aperture complexity, scaling
                  with D^α (e.g., D^2.3) to account for structural
                  and control challenges.
    - C_fixed: A fixed programmatic floor for the bus, GNC, software, etc.
    - M_o: An overhead and margin multiplier applied to all CAPEX.

    OPEX includes:
    - C_energy: Total cost of electrical energy for the mission.
    - C_time: Total cost of mission operations, scaling with "laser-on" time.

    Args:
        x (dict): The dictionary of input design parameters.
        c (dict): A dictionary of all cost coefficients
                  (e.g., 'k_area_$per_m2', 'k_laser_$per_kW', 'program_floor_$', etc.).
        Cm_use (float, optional): A specific C_m value to pass to
                                  design_metrics. Defaults to None.

    Returns:
        float: The total estimated mission cost in dollars.
    """
    met = design_metrics(x, Cm_override=(Cm_use if Cm_use is not None else x.get('cm_min')))

    D = x['D']
    area_m2 = math.pi * (D/2)**2

    # Base pieces
    C_optics = c['k_area_$per_m2'] * area_m2

    P_elec_avg_kW = met['P_elec_avg_W'] / 1e3
    C_laser = max(c['k_laser_$per_kW'] * P_elec_avg_kW, c['laser_capex_floor_$'])

    # Nonlinear aperture term + fixed program floor
    C_aperture = c['cap_ap_coef_$per_m_pow_alpha'] * (D ** c['cap_ap_alpha'])
    C_fixed    = c['program_floor_$']

    # Apply overheads/margins to CAPEX
    C_capex = (C_fixed + C_optics + C_laser + C_aperture) * c['overhead_mult']

    # OpEx (energy + ops-time)
    C_energy = c['price_per_kWh'] * (met['E_total_J'] / 3.6e6)
    C_time   = c['k_time_$per_hr'] * (met['mission_time_s'] / 3600.0)

    return float(C_capex + C_energy + C_time)


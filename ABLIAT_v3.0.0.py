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

# ---- plotting helpers (smaller, centered figures) ----
FIGSIZE = (5.0, 3.5)
FIGSIZE_DV = (FIGSIZE[0]*2.0, FIGSIZE[1]*2.0)   # ~2× bigger just for Δv–t
#FIGSIZE_BIG = (FIGSIZE[0]*1.5, FIGSIZE[1]*1.5)
FIGSIZE_BIG = FIGSIZE
PYplotOpts = dict(clear_figure=True, use_container_width=False)
# ---- Figure formatting helpers ----
DEFAULT_DPI = 150
LEGEND_BBOX = (1.02, 0.5)   # to the right, vertically centered
RIGHT_MARGIN = 0.75         # leave space on the right for the legend

# SciPy optimiser (falls back to random search if missing)
_SCIPY_OK = False
with contextlib.suppress(Exception):
    from scipy.optimize import differential_evolution  # type: ignore
    _SCIPY_OK = True


# ---- HELPER FUNCTIONS ----
# =====================
# Overspill helpers (centered top-hat, normal incidence)
# =====================
def j_to_kwh(EJ: float) -> float:
    return EJ / 3.6e6

def obj_diameter_from_mu(m_kg: float, mu_kg_m2: float) -> float:
    """Projected-equivalent diameter [m] from mass and areal mass density μ (kg/m²)."""
    if mu_kg_m2 <= 0:
        return 0.0
    A_proj = m_kg / mu_kg_m2                 # m²
    return 2.0 * math.sqrt(max(A_proj, 0.0) / math.pi)

def obj_diameter_from_density(m_kg: float, rho_kg_m3: float) -> float:
    """Spherical equivalent diameter [m] from mass and bulk density ρ (kg/m³)."""
    if rho_kg_m3 <= 0:
        return 0.0
    # m = ρ * (π/6) * d^3  =>  d = (6m/(πρ))^(1/3)
    return (6.0 * m_kg / (math.pi * rho_kg_m3)) ** (1.0 / 3.0)

def eta_geom_tophat_centered(d_obj_m: float, d_spot_m: float) -> float:
    """Geometric capture fraction for a centered uniform spot on a circular target."""
    if d_spot_m <= 0:
        return 1.0
    r = max(d_obj_m, 0.0) / d_spot_m
    return 1.0 if r >= 1.0 else r * r  # area ratio

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

def E_pulse_needed_for_F(Fmin_J_cm2: float, d_m: float, eta_abs: float) -> float:
    """
    Calculates the minimum *optical* pulse energy required to meet a fluence target.

    This is the reverse calculation for fluence (E = Φ * A). It determines
    the *incident* optical energy (in Joules) needed to achieve a specific
    *absorbed* fluence (Fmin_J_cm2) on a given spot size (d_m),
    accounting for the target's absorption efficiency (eta_abs).

    Args:
        Fmin_J_cm2 (float): The target *absorbed* fluence in J/cm².
        d_m (float): The on-target spot diameter in meters.
        eta_abs (float): The absorption efficiency of the target (0.0 to 1.0).

    Returns:
        float: The minimum *incident* optical pulse energy (in Joules)
               required to meet the fluence target.
    """
    return (Fmin_J_cm2 * math.pi * (d_m*100.0)**2) / (4.0 * max(eta_abs, 1e-9))

def enforce_min_fluence(x: dict, req: dict, P_limit_W: float | None = None) -> dict:
    """
    Validates and enforces fluence and power constraints on a candidate design.

    This function is a critical part of the optimizer's objective function.
    It performs three main checks:
    1.  Calculates the minimum pulse energy (E_need) required to meet the
        F_min_J_cm2 constraint for the candidate's spot size.
    2.  Checks if this E_need is even possible, given the E_max and
        average power (P_limit_W) constraints.
    3.  Compares the optimizer's proposed energy (E_proposed) to E_need.
        If E_proposed is too low, it's "snapped" up to E_need.
        If E_proposed is too high (violating power caps), the design
        is marked as infeasible.

    It also stores E_need in the x dictionary for use in other functions.

    Args:
        x (dict): The candidate design dictionary from the optimizer.
        req (dict): A dictionary of all constraint requirements.
        P_limit_W (float, optional): An *additional* average power cap,
                                     often passed from the objective
                                     function. Defaults to None.

    Returns:
        dict: The updated design dictionary (x), either with a valid,
              enforced E_pulse or with an __infeasible__ flag set to True.
    """
    # spot size (meters -> cm)
    d_m = spot_diameter(
        x['L'], x['D'], x['lam'], x['M2'], x['jitter_urad'],
        x.get('spot_model', 'gaussian_1e2')
    )
    d_cm  = 100.0 * d_m
    A_cm2 = 0.25 * np.pi * d_cm**2

    # efficiencies
    eta_abs  = float(x['eta_abs'])
    eta_geom = float(x.get('eta_geom', 1.0))  

    # min absorbed fluence from UI/config
    Phi_min = float(req['F_min_J_cm2'])  
    Phi_req_inc = Phi_min / max(eta_abs * eta_geom, 1e-12)
    E_need = Phi_req_inc * A_cm2  # Joules (optical)
    x['__E_need_for_Fmin__'] = E_need

    # respect input E bounds
    E_min = float(req['E_min_J'])
    E_max = float(req['E_max_J']) 

    # power cap: E <= P_limit_W / f
    if P_limit_W is not None:
        f = float(x['f_rep'])
        if f > 0:
            E_cap_power = P_limit_W / f
            E_max_eff = min(E_max, E_cap_power)
        else:
            E_max_eff = E_max
    else:
        E_max_eff = E_max

    # A) Check if Φ_min is even possible with this spot/power cap
    if E_need > E_max_eff + 1e-12:
        x['__infeasible__'] = True
        x['__why__'] = (f"Φ_min requires ~{E_need:.0f} J (for this spot) but cap allows ≤{E_max_eff:.0f} J.")
        return x

    # B) Get the optimiser's proposed energy
    E_proposed = float(x['E_pulse']) 

    # C) Compare proposed energy to the minimum needed
    if E_proposed >= E_need:
        # Proposed energy is valid (it's >= Φ_min).
        # We must re-check it against the *effective* power cap.
        if E_proposed > E_max_eff + 1e-12:
            x['__infeasible__'] = True
            x['__why__'] = (f"Proposed E={E_proposed:.0f} J exceeds power cap E_max_eff={E_max_eff:.0f} J "
                            f"(P_limit={P_limit_W} W / f={f:.1f} Hz).")
            return x
        
        # Valid: Optimiser's choice is respected.
        x['E_pulse'] = E_proposed
        x['__E_set_by__'] = 'optimiser'
    else:
        # Proposed energy is TOO LOW to meet Φ_min.
        # Force it to use the minimum required energy (E_need).
        # Already know E_need <= E_max_eff from check A.
        x['E_pulse'] = float(np.clip(E_need, E_min, E_max_eff))
        x['__E_set_by__'] = 'fluence_min'

    return x

def run_opt(cm_choice: str, bounds, x_template, req, cost_cfg, allowed_taus, T_eng: float):
    """
    Runs the optimisation process using SciPy's differential evolution (or a
    random search fallback).

    This wrapper handles the setup, execution, and result parsing of the optimisation.
    It tries to find the optimal combination of design variables (D, L, E_pulse,
    tau, f_rep) that minimises the objective function cost.

    After optimisation, it calculates the final detailed metrics for the best
    found solution, including power bookkeeping and duty factors.

    Args:
        cm_choice (str): Which C_m mode to optimise for ('min', 'max', or 'fix').
        bounds (list): List of tuples [(min, max), ...] defining the search space
                       for each design variable.
        x_template (dict): Dictionary of static system parameters.
        req (dict): Dictionary of constraint requirements (max power, min fluence, etc.).
        cost_cfg (dict): Dictionary of cost model coefficients.
        allowed_taus (list): List of discrete pulse durations allowed.
        T_eng (float): Engagement window duration in seconds (used for duty factor calculation).

    Returns:
        tuple: (x_opt, met, Cm_use, cost)
            - x_opt (dict): The optimised design variables.
            - met (dict): Detailed performance metrics for the optimal design.
            - Cm_use (float): The C_m value used for the final calculation.
            - cost (float): The final minimized cost value.
    """
    if _SCIPY_OK and len(allowed_taus) > 0:
        result = differential_evolution(
            objective, bounds=bounds,
            args=(x_template, req, cost_cfg, allowed_taus, cm_choice, T_eng),
            strategy='best1bin', maxiter=120, popsize=25, tol=1e-6,
            polish=True, disp=False,
            workers=1,
            seed = (3 if cm_choice=='min' else (4 if cm_choice=='max' else 5))
        )
        vec = result.x
    else:
        rng = np.random.default_rng(3 if cm_choice=='min' else (4 if cm_choice=='max' else 5))
        best_val, best_vec = float('inf'), None
        for _ in range(6000):
            D = rng.uniform(req['D_min_m'], req['D_max_m'])
            L = rng.uniform(req['L_min_m'], req['L_max_m'])
            E = rng.uniform(req['E_min_J'], req['E_max_J'])
            tau = float(rng.choice(allowed_taus))
            f = 10**rng.uniform(math.log10(req['f_min_Hz']), math.log10(req['f_max_Hz']))
            val = objective([D, L, E, math.log10(tau), math.log10(f)],
                            x_template, req, cost_cfg, allowed_taus, cm_choice, T_eng)
            if val < best_val:
                best_val, best_vec = val, [D, L, E, math.log10(tau), math.log10(f)]
        vec = np.array(best_vec)

    D, L, E_pulse, log_tau, log_f = vec
    tau_cont = 10**log_tau
    tau = allowed_taus[np.argmin([abs(math.log10(tau_cont) - math.log10(t)) for t in allowed_taus])]
    f_rep = 10**log_f

    x_opt = dict(x_template)
    x_opt.update({'D': float(D), 'L': float(L), 'E_pulse': float(E_pulse), 'tau': float(tau), 'f_rep': float(f_rep)})
    x_opt = enforce_min_fluence(x_opt, req)

    if cm_choice == 'min':
        Cm_use = x_opt['cm_min']
    elif cm_choice == 'max':
        Cm_use = x_opt['cm_max']
    else:  # 'fix'
        Cm_use = x_opt['cm_fixed']


    met = design_metrics(x_opt, Cm_override=Cm_use)

    met = design_metrics(x_opt, Cm_override=Cm_use)

    # --- Feasibility check for T_eng removed ---
    met["__infeasible__"] = False
    cost = cost_model(x_opt, cost_cfg, Cm_use=Cm_use)

    # power bookkeeping
    met["P_burst_W"] = met["P_avg_W"]

    # Duty factor from engagement window T_eng (passed into run_opt) vs required on-time
    met["T_eng_s"]     = float(T_eng)
    met["duty_factor"] = min(1.0, met["mission_time_s"] / max(met["T_eng_s"], 1.0))

    # Time-averaged optical power over the on/off window
    met["P_timeavg_W"] = met["P_burst_W"] * met["duty_factor"]

    return x_opt, met, Cm_use, cost

def objective(vec, template_x: dict, req: dict, cost_cfg: dict,
              allowed_taus: list[float], cm_choice: str, T_eng: float) -> float:
    """
    Cost function for the differential_evolution optimiser.

    This function takes a vector of design variables from the optimiser,
    builds a complete system configuration, and checks it against a series
    of hard physical and engineering constraints.

    If a candidate solution is infeasible (violates any constraint), it
    returns a massive penalty (1e12). If it is feasible, it calculates
    and returns the total mission cost using the cost_model.

    Args:
        vec (np.ndarray): The vector of optimisation variables [D, L, E_pulse, log_tau, log_f]
                          from the differential_evolution algorithm.
        template_x (dict): A dictionary of all static system parameters.
        req (dict): A dictionary of all constraint requirements (min/max values).
        cost_cfg (dict): A dictionary of cost coefficients for the cost model.
        allowed_taus (list): A list of discrete pulse durations (in seconds) to which
                             the optimiser's continuous log_tau value will be "snapped".
        cm_choice (str): A string ('min', 'max', or 'fix') indicating which C_m
                         value to use for the calculation.
        T_eng (float): The available engagement time (in seconds). This is passed in
                       but no longer used as a hard constraint.

    Returns:
        float: The total mission cost if the design is feasible, or a large
               penalty (1e12) if it is infeasible.
    """
    D, L, E_pulse, log_tau, log_f = vec
    tau_cont = 10**log_tau
    tau = allowed_taus[np.argmin([abs(math.log10(tau_cont) - math.log10(t)) for t in allowed_taus])]
    f_rep = 10**log_f

    x = dict(template_x)
    x.update({'D': float(D), 'L': float(L), 'E_pulse': float(E_pulse), 'tau': float(tau), 'f_rep': float(f_rep)})
    x = enforce_min_fluence(x, req)
    if x.get('__infeasible__', False):
        return 1e12

    if cm_choice == 'min':
        Cm_use = x.get('cm_min')
    elif cm_choice == 'max':
        Cm_use = x.get('cm_max')
    elif cm_choice == 'fix':
        Cm_use = x.get('cm_fixed')
    else:
        raise ValueError(f"unknown cm_choice: {cm_choice}")

    met = design_metrics(x, Cm_override=Cm_use)
    violations = []

    # NEW: Thermal/Heat Dissipation Constraint
    # Get wall-plug efficiency (default to 20% if not found)
    eta_wall = max(x.get('eta_wallplug', 0.2), 1e-3) 

    # Calculate the waste heat the *laser is producing*
    # P_waste = P_elec - P_optical = (P_avg/eta) - P_avg
    P_waste_produced = met['P_avg_W'] * ((1.0 / eta_wall) - 1.0)
    
    # Get the radiator's maximum capacity (calculated in Step 1)
    P_waste_max_allowed = req['P_waste_max_W'] 

    # Add violation if the produced heat exceeds what the radiator can handle
    violations.append(max(0.0, (P_waste_produced - P_waste_max_allowed) / max(P_waste_max_allowed, 1.0)))

    # Spot, fluence, range, aperture, energy, power constraints
    violations.append(max(0.0, (met['spot_diameter_m'] - req['d_max_m']) / req['d_max_m']))
    violations.append(max(0.0, (req['F_min_J_cm2'] - met['fluence_J_cm2']) / req['F_min_J_cm2']))
    violations.append(max(0.0, (met['fluence_J_cm2'] - req['F_max_J_cm2']) / max(req['F_max_J_cm2'], 1e-9)))
    violations.append(max(0.0, (req['L_min_m'] - x['L']) / max(req['L_min_m'],1.0)))
    violations.append(max(0.0, (x['L'] - req['L_max_m']) / max(req['L_max_m'],1.0)))
    violations.append(max(0.0, (req['D_min_m'] - x['D']) / req['D_min_m']))
    violations.append(max(0.0, (x['D'] - req['D_max_m']) / req['D_max_m']))
    violations.append(max(0.0, (req['E_min_J'] - x['E_pulse']) / max(req['E_min_J'],1.0)))
    violations.append(max(0.0, (x['E_pulse'] - req['E_max_J']) / max(req['E_max_J'],1.0)))
    violations.append(max(0.0, (met['P_peak_W'] - req['P_peak_max_W']) / max(req['P_peak_max_W'],1.0)))
    violations.append(max(0.0, (met['P_avg_W']  - req['P_avg_max_W'])  / max(req['P_avg_max_W'],1.0)))

    # Engagement-time feasibility (baseline target must fit in window)
    T_need = met['mission_time_s']  # for m_target and dv_target


    V = sum(violations)
    if V > 0:
        return 1e12 * V

    return cost_model(x, cost_cfg, Cm_use=Cm_use)

def integrate_distribution_energy(
    df, Cm_Ns_per_J, Epulse_J, eta_abs,
    rho_kg_m3=2700.0, hp_km=300.0, xi_geom=1.0,
    count_col="N_total_bin", alt_col="Altitude_km", diam_col="Diameter_m"
):
    """
    Calculates the total energy and pulses required to de-orbit an entire debris population.

    This function iterates through a debris distribution DataFrame (e.g., from ESA
    MASTER), treating each row as a 'bin' of objects. For each bin, it:
    1. Calculates the mass of a single object (assuming an aluminium sphere).
    2. Calculates the required Δv to lower its perigee from its current
       altitude to the hp_km target.
    3. Calculates the number of laser pulses (n_p) needed to achieve this Δv.
    4. Multiplies these per-object needs by the number of objects (n) in that bin.

    Returns:
        dict: A dictionary containing the grand totals for the entire population
              (total_objects, total_pulses, total_optical_J, total_absorbed_J)
              and a detailed DataFrame (breakdown) with the per-bin calculations.
    """
    rows, total_pulses, total_objects = [], 0.0, 0.0
    for _, r in df.iterrows():
        n = float(r[count_col])
        if n <= 0: continue
        h_km = float(r[alt_col]); d_m = float(r[diam_col])
        dv = dv_to_perigee_300km(h_km, hp_km=hp_km)
        m  = rho_kg_m3 * (math.pi / 6.0) * (d_m ** 3)          # sphere
        E_eff = max(eta_abs * Epulse_J, 1e-12)
        Ipp   = Cm_Ns_per_J * E_eff * xi_geom                   # impulse per pulse
        n_p   = (m * dv) / max(Ipp, 1e-18)
        total_pulses  += n_p * n
        total_objects += n
        rows.append({
            "Altitude_km": h_km, "Diameter_m": d_m, "Count": n, "dv_m_s": dv, "mass_kg": m,
            "pulses_per_obj": n_p,
            "optical_energy_per_obj_J": n_p * Epulse_J,
            "absorbed_energy_per_obj_J": n_p * Epulse_J * eta_abs,
        })
    breakdown = pd.DataFrame(rows)
    return {
        "total_objects": total_objects,
        "total_pulses": total_pulses,
        "total_optical_J": total_pulses * Epulse_J,
        "total_absorbed_J": total_pulses * Epulse_J * eta_abs,
        "breakdown": breakdown
    }

def new_fig(figsize):
    """Create a fresh figure/axes with consistent DPI."""
    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)
    return fig, ax

def render_with_right_legend(fig, ax, right=RIGHT_MARGIN):
    """Place legend on the right, drop helper entries, render w/o container auto-shrink."""
    handles, labels = ax.get_legend_handles_labels()
    kept = [(h, l) for h, l in zip(handles, labels) if l and l != "_nolegend_"]
    if kept:
        handles, labels = zip(*kept)
        ax.legend(handles, labels, loc="center left",
                  bbox_to_anchor=LEGEND_BBOX, borderaxespad=0.0, frameon=True)
    fig.subplots_adjust(right=right)
    st.pyplot(fig, use_container_width=False)
    

def centered_pyplot(fig):
    """Render a pyplot figure centered on the page (via 3-column layout)."""
    left, mid, right = st.columns([1, 6, 1])
    with mid:
        st.pyplot(fig, **PYplotOpts)


# =====================
# Streamlit UI
# =====================

st.set_page_config(page_title="Laser Debris Optimiser", layout='wide')
st.title("ABLIAT - Ablation Laser Impulse Analysis Tool (v3.0.0)")

with st.sidebar:
    st.header("Scenario & beam physics")
    lam_options = { # Not polished - Use single lambda
        "UV 355 nm": 355e-9, 
        #"Nd:YAG 532 nm": 532e-9, 
        #"Nd:YAG 1064 nm": 1064e-9
    }
    lam_labels = st.multiselect(
        "Wavelengths (λ) to Test", 
        list(lam_options.keys()), 
        default=["UV 355 nm"]
    )
    lam_choices = [lam_options[label] for label in lam_labels]

    spot_model = st.selectbox(
        "Spot definition",
        ["gaussian_1e2","airy_first_null","gaussian_fwhm","airy_fwhm"],
        index=0,
        help="Choose the diameter convention used throughout (impacts fluence)"
    )

    M2 = st.number_input("Beam quality M²", min_value=1.0, max_value=5.0, value=2.0, step=0.05)
    jitter = st.number_input("Pointing jitter (µrad RMS)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    eta_abs = st.slider("Absorption at target η_abs", 0.05, 1.0, 1.0, 0.05)

    st.header("Target & maneuver")
    m_target = st.number_input("Target mass m (kg)", min_value=0.01, max_value=10000.0, value=1.0, step=0.1)
    dv_target = st.number_input("Required Δv (m/s)", min_value=0.01, max_value=1000.0, value=140.0, step=1.0)
    use_distribution = st.checkbox(
        "Use debris distribution (CSV roll-up)",
        value=False,
        help="Keep target mass for optimisation. If checked, upload a DRAMA CSV to roll up total energy."
    )


    if use_distribution:
        import pandas as pd
        dist_csv = st.file_uploader("Upload DRAMA CSV", type=["csv"], key="dist_csv")

        xi_geom = st.slider("Geometry efficiency ξ_geom", 0.30, 1.00, 1.00, 0.05, key="xi_geom_rollup")
        hp_km   = st.number_input("Target perigee [km]", 200.0, 500.0, 300.0, 10.0, key="hp_rollup")
        #eta_wall = st.slider("Wall-plug efficiency (optical←electrical)", 0.01, 1.0, 1.0, 0.01, key="wall_eff_rollup")

        rho_bulk_use = 2700.0

        if dist_csv is not None:
            st.session_state['drama_df'] = pd.read_csv(dist_csv)
            st.session_state['drama_filename'] = dist_csv.name
            st.success("DRAMA CSV loaded.")
        else:
            st.session_state.pop('drama_df', None)
            st.session_state.pop('drama_filename', None)

    st.header("C_m settings")

    cm_min  = st.number_input("C_m min (N·s/J)",  value=1e-4,  format="%.6f")
    cm_max  = st.number_input("C_m max (N·s/J)",  value=1e-3,  format="%.6f")
    cm_fixed  = st.number_input("Fixed C_m (N·s/J)", value=1.4e-5, format="%.6f")

    # simple validation
    if cm_min <= 0 or cm_max <= 0 or cm_fixed <= 0:
        st.warning("All C_m values must be > 0.")
    if cm_min > cm_max:
        st.warning("C_m min must be ≤ C_m max.")


    st.header("Constraints")
    Dmin, Dmax = st.slider("Aperture D (m)", 0.1, 5.0, (3.5, 5.0), 0.1)
    Lrange_km = st.slider("Operating range L (km)", 10.0, 500.0, (200.0, 250.0), 10.0)
    Lmin, Lmax = [v * 1e3 for v in Lrange_km]
    dmax_cm = st.number_input("Max spot d_max (cm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    dmax_m = dmax_cm / 100.0
    Fmin = st.number_input("Min fluence Φ_min (J/cm²)", min_value=0.1, max_value=40.0, value=0.85, step=0.1)
    Fmax = st.number_input("Max fluence Φ_max (J/cm²)", min_value=0.2, max_value=50.0, value=0.95, step=0.1,
                      help="Ceiling cap for fluence to avoid plasma shielding.")

    st.header("Laser bounds & limits")
    Emin, Emax = st.slider("Pulse energy bounds E (J)", 1.0, 1000.0, (75.0, 500.0), 1.0)
    fr_min, fr_max = st.slider("Rep-rate bounds f (Hz)", 1.0, 300.0, (35.0, 200.0))
    Pavg_max = st.number_input("Max average optical power (W)", min_value=1e3, max_value=2e10, value=2e5, step=1e3)
    Ppeak_max = st.number_input("Max peak power (W)", min_value=1e6, max_value=1e18, value=1e16, step=1e6)

    st.header("Efficiencies")
    eta_wall = st.slider("Wall-plug efficiency η_wall", 0.0, 1.0, 0.2, 0.01,
                         help="Optical avg power / electrical avg power")

    st.header("Overheating Constraints")
    A_rad = st.number_input("Radiator Area (m^2)", 1.0, 100.0, 18.0, 1.0,
                            help="Total area of the spacecraft's radiators.")
    T_rad = st.number_input("Radiator Temp (K)", 250.0, 750.0, 420.0, 5.0,
                            help="Operating temperature of the radiators.")
    epsilon = st.slider("Radiator Emissivity (ε)", 0.5, 1.0, 0.85, 0.01,
                        help="Effectiveness of the radiator surface (0.8-0.9 is common).")

    # ---------- Overspill geometry ----------
    st.header("Overspill geometry")
    overspill_model = st.radio(
        "Geometry model",
        ["None (legacy)", "Areal mass density μ", "Mass + bulk density ρ"],
        index=0,
        help="How much of each pulse actually lands on-target when the spot is larger than the object."
    )
    mu_areal = None
    rho_bulk = None
    if overspill_model == "Areal mass density μ":
        mu_areal = st.number_input(
            "Areal mass density μ (kg/m²)",
            min_value=0.1, max_value=1_000.0, value=10.0, step=0.5
        )
    elif overspill_model == "Mass + bulk density ρ":
        rho_bulk = st.number_input(
            "Bulk density ρ (kg/m³) [spherical target]",
            min_value=100.0, max_value=25_000.0, value=2700.0, step=50.0
        )

    # ---------- Engagement window ----------
    st.header("Engagement window")
    eng_mode = st.radio(
        "Window source",
        ["From range + relative v", "Custom time (s)"],
        index=0,
        help="Time available to lase during a single pass/approach."
    )

    v_rel_kms = None  # keep available for plotting later
    if eng_mode == "From range + relative v":
        v_rel_kms = st.number_input("Relative speed (km/s)", 0.1, 20.0, 15.0, 0.5)
        # T_eng here is only for pre-checks; plots will use L_opt / v_rel
        T_eng = max((Lmax - Lmin) / max(v_rel_kms * 1e3, 1e-9), 0.0)
    else:
        T_eng = st.number_input("Custom engagement time T_eng (s)", 0.1, 3000.0, 20.0, 1.0)


    st.header("Allowed pulse durations τ")
    allowed_taus = st.multiselect(
        "Discrete τ values",
        [1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
        default=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
        format_func=lambda t: ("100 ps" if math.isclose(t,1e-10)
                               else "1 ns" if math.isclose(t,1e-9)
                               else "10 ns" if math.isclose(t,1e-8)
                               else "100 ns" if math.isclose(t,1e-7)
                               else "1 µs")
    )

    st.header("Cost model")
    k_area   = st.number_input("Optics CAPEX $/m²", 0.0, 5e6, 3e5, 1e4)
    k_laser  = st.number_input("Laser CAPEX $/kW (avg)", 0.0, 10e6, 2e6, 1e3)
    price_kWh= st.number_input("Energy price $/kWh", 0.0, 10.0, 1.75, 0.01)
    k_time   = st.number_input("Ops cost $/hour", 0.0, 1e6, 2e3, 10.0)

    program_floor     = st.number_input("Programmatic floor CAPEX $", value=1.2e8, step=1e7, format="%.0f")
    laser_capex_floor = st.number_input("Laser CAPEX floor $",         value=4.0e7, step=1e7, format="%.0f")
    cap_ap_coef       = st.number_input("Aperture scaling coef $/(m^α)", value=1.20e6, step=1e5, format="%.0f")
    cap_ap_alpha      = st.number_input("Aperture scaling exponent α", value=2.3, step=0.1)
    overhead_mult     = st.number_input("Overheads & Margin multiplier", value=1.30, step=0.05, format="%.2f")


    do_opt = st.button("Optimise configuration")

# --- Feasibility diagnostic (quick pre-check) ---
d_best = spot_diameter(Lmin, Dmax, lam_choices[0], M2, jitter, model=spot_model)   # [m]
E_need = E_pulse_needed_for_F(Fmin, d_best, eta_abs)                           # [J]
Pavg_min = E_need * fr_min                                                     # [W] optical

if E_need > Emax:
    st.warning("Infeasible: E_needed exceeds E_max. Raise E_max, shrink spot (↑D_max or ↓L_min), "
               "or relax F_min / increase η_abs.")
if Pavg_min > Pavg_max:
    st.warning("Infeasible: minimum average optical power exceeds P_avg_max. "
               "Lower f_min or raise P_avg_max (or both).")

# Time-feasibility heads-up (optimistic bound)
if (E_need <= Emax) and (Pavg_min <= Pavg_max):
    slope_best = (cm_max * eta_abs * Emax * fr_max) / max(m_target, 1e-30)
    if slope_best * T_eng < dv_target:
        st.warning("Time-feasibility risk: even best-case (C_m,max, E_max, f_max) "
                   "may not reach Δv within the engagement window.")

# Templates
x_template = {
    'D': 1.0, 'L': max(100e3, 100e3), 'lam': 355e-9,
    'E_pulse': 300.0, 'tau': 100e-12, 'f_rep': 10.0,
    'eta_abs': eta_abs, 'M2': M2, 'jitter_urad': jitter,
    'm_target': m_target, 'dv_target': dv_target,
    'cm_fixed': cm_fixed,
    'cm_min': cm_min, 'cm_max': cm_max,
    'eta_wallplug': eta_wall, 'spot_model': spot_model,
}

# --- Calculate Radiator Capacity ---
# P_rad = ε * σ * A * T^4 (in Watts)
P_waste_max_W = epsilon * STEFAN_BOLTZMANN * A_rad * (T_rad**4)
st.sidebar.info(f"Radiator Capacity: {P_waste_max_W / 1e3:.1f} kW") # Show user the result
# ---

req = {
    'd_max_m': dmax_m, 'F_min_J_cm2': Fmin,
    'F_max_J_cm2': Fmax,
    'L_min_m': Lmin, 'L_max_m': Lmax,
    'D_min_m': Dmin, 'D_max_m': Dmax,
    'E_min_J': Emin, 'E_max_J': Emax,
    'f_min_Hz': fr_min, 'f_max_Hz': fr_max,
    'P_peak_max_W': Ppeak_max, 'P_avg_max_W': Pavg_max,
    'P_waste_max_W': P_waste_max_W
}

cost_cfg = {
    'k_area_$per_m2': k_area,
    'k_laser_$per_kW': k_laser,
    'price_per_kWh': price_kWh,
    'k_time_$per_hr': k_time,
    'program_floor_$': program_floor,
    'laser_capex_floor_$': laser_capex_floor,
    'cap_ap_coef_$per_m_pow_alpha': cap_ap_coef,
    'cap_ap_alpha': cap_ap_alpha,
    'overhead_mult': overhead_mult,
}


#save Cm ranges
Cm_best = None
Cm_cons = None
# =============
# Optimisation (three cases: Cm_min, Cm_max, and Cm_fixed)
# =============
if do_opt:
    for lam_choice in lam_choices:
        st.subheader(f"OPTIMISATION FOR {int(lam_choice*1e9)} nm")
        
        # Update the template with the current wavelength for this loop
        x_template['lam'] = lam_choice

        bounds = [
            (Dmin, Dmax),                               # D
            (Lmin, Lmax),                               # L
            (Emin, Emax),                               # E_pulse
            (math.log10(min(allowed_taus)), math.log10(max(allowed_taus))),  # log10(tau)
            (math.log10(fr_min), math.log10(fr_max)),   # log10(f)
        ]

        x_min, met_min, Cm_min_used, cost_min = run_opt('min', bounds, x_template, req, cost_cfg, allowed_taus, T_eng)
        x_max, met_max, Cm_max_used, cost_max = run_opt('max', bounds, x_template, req, cost_cfg, allowed_taus, T_eng)
        x_fix, met_fix, Cm_fix_used, cost_fix = run_opt('fix', bounds, x_template, req, cost_cfg, allowed_taus, T_eng)
        
        Cm_best = Cm_max_used
        Cm_cons = Cm_min_used

        st.subheader("Optimisation summary table")

        metrics = [
        "Aperture D (m)", "Range L (km)", "Spot d (cm)", "Fluence Φ (J/cm²)",
        "Pulse energy E (J)", "τ (ns)", "f (Hz)",
        "P_burst (kW)", "P_avg (kW)", "Pulses N", "E_total (kWh)"
        ]

        def fmt_row(x, met):
            return [
                f"{x['D']:.3f}",
                f"{x['L']/1e3:.1f}",
                f"{met['spot_diameter_m']*100:.2f}",
                f"{met['fluence_J_cm2']:.2f}",
                f"{x['E_pulse']:.1f}",
                f"{x['tau']*1e9:.1f}",
                f"{x['f_rep']:.1f}",
                f"{met['P_burst_W']/1e3:.2f}",    # kW
                f"{met['P_timeavg_W']/1e3:.2f}",  # kW
                f"{met['N_pulses']:.2e}",
                f"{j_to_kwh(met['E_total_J']):.3f}",
            ]

        col_worst = fmt_row(x_min, met_min)
        col_best  = fmt_row(x_max, met_max)
        col_fix = fmt_row(x_fix, met_fix)
        cols = [
            f"Conservative (C_m={Cm_min_used:.1e})",
            f"Best-case (C_m={Cm_max_used:.1e})",
            f"Fixed (C_m={Cm_fix_used:.1e})",
        ]
        df = pd.DataFrame({cols[0]: col_worst, cols[1]: col_best, cols[2]: col_fix}, index=metrics)
        st.table(df)

        # Save data 
        if use_distribution and 'drama_filename' in st.session_state:
            try:
                lam_nm_str = f"_{int(lam_choice*1e9)}nm"
                base_name = st.session_state['drama_filename'].split('.csv')[0]
                summary_filename = f"{base_name}_optimisation_summary{lam_nm_str}.csv"
                df.to_csv(summary_filename)
                st.success(f"Saved optimisation summary to {summary_filename}")
            except Exception as e:
                st.warning(f"Could not save summary CSV: {e}")


        x_lo = x_min
        x_hi = x_max
        eta_use = x_lo['eta_abs']
        dv_goal = x_lo['dv_target']
        
        st.divider()

# =========================
# Debris Distribution Results
# =========================

# Use the two optimised designs if available; else fall back to template
x_lo = x_min if 'x_min' in locals() else x_template   # worst-case Cm design
x_hi = x_max if 'x_max' in locals() else x_template   # best-case  Cm design
x_fix_use = x_fix if 'x_fix' in locals() else x_template # Fixed-Cm design

eta_use = x_lo['eta_abs']
dv_goal = x_lo['dv_target']  # same in both designs

# Design-specific engagement windows for plotting
# If the window comes from L-range + v_rel, use 0 -> L_opt for each design.
if (v_rel_kms is not None):
    T_eng_lo = x_lo['L'] / max(v_rel_kms * 1e3, 1e-9)
    T_eng_hi = x_hi['L'] / max(v_rel_kms * 1e3, 1e-9)
else:
    T_eng_lo = T_eng_hi = T_eng

# --- Spot sizes (m) for both designs (needed for overspill) ---
met_lo = design_metrics(x_lo, Cm_override=x_lo.get('cm_min'))
met_hi = design_metrics(x_hi, Cm_override=x_hi.get('cm_max'))
d_spot_lo_m = met_lo['spot_diameter_m']
d_spot_hi_m = met_hi['spot_diameter_m']

# --- Overspill capture functions and threshold masses for plotting ---
def make_eta_fn(model_label: str, d_spot_m: float):
    if model_label == "None (legacy)":
        return (lambda m: 1.0), None
    if model_label == "Areal mass density μ" and (mu_areal is not None):
        # Threshold mass where d_obj == d_spot:  m_th = μ * π * (d_spot/2)^2
        m_thresh = mu_areal * math.pi * (d_spot_m * 0.5)**2
        def eta(m):  # m in kg
            d_obj = obj_diameter_from_mu(m, mu_areal)
            return eta_geom_tophat_centered(d_obj, d_spot_m)
        return eta, m_thresh
    if model_label == "Mass + bulk density ρ" and (rho_bulk is not None):
        # Threshold mass:  m_th = ρ * (π/6) * d_spot^3
        m_thresh = rho_bulk * (math.pi / 6.0) * (d_spot_m ** 3)
        def eta(m):
            d_obj = obj_diameter_from_density(m, rho_bulk)
            return eta_geom_tophat_centered(d_obj, d_spot_m)
        return eta, m_thresh
    # Fallback
    return (lambda m: 1.0), None

eta_lo, mth_lo = make_eta_fn(overspill_model, d_spot_lo_m)
eta_hi, mth_hi = make_eta_fn(overspill_model, d_spot_hi_m)

# ---------- Distribution roll-up results (main pane, AFTER optimisation) ----------
if use_distribution:
    st.subheader("Total energy to process distribution → perigee 300 km")
    if 'drama_df' not in st.session_state:
        st.info("Upload a DRAMA CSV in the sidebar to enable distribution roll-up.")
    else:
        dist_df = st.session_state['drama_df']

        # Recompute metrics here to guarantee 'E_use' (threshold-enforced pulse energy) exists
        met_lo_roll = design_metrics(x_lo, Cm_override=x_lo.get('cm_min'))
        met_hi_roll = design_metrics(x_hi, Cm_override=x_hi.get('cm_max'))
        met_fix_roll = design_metrics(x_fix_use, Cm_override=cm_fixed)
        Epulse_fix   = met_fix_roll.get('E_use', x_fix_use.get('E_pulse', 0.0))
        Epulse_lo = met_lo_roll.get('E_use', x_lo.get('E_pulse', 0.0))
        Epulse_hi = met_hi_roll.get('E_use', x_hi.get('E_pulse', 0.0))
        eta_abs_roll = eta_use if 'eta_use' in locals() else x_lo.get('eta_abs', 1.0)

        designs = [
            ("Conservative (C_m = min)", x_lo['cm_min'], Epulse_lo),
            ("Best-case (C_m = max)",    x_hi['cm_max'], Epulse_hi),
            (f"Fixed (C_m = {cm_fixed:.0e})", cm_fixed, Epulse_fix),
        ]

        # --- Build per-design results with COST breakdown for the distribution roll-up ---
        results = []

        design_triplets = [
            ("Conservative (C_m = min)", x_lo, met_lo_roll, x_lo['cm_min'], Epulse_lo),
            ("Best-case (C_m = max)",    x_hi, met_hi_roll, x_hi['cm_max'], Epulse_hi),
            (f"Fixed (C_m = {cm_fixed:.0e})", x_fix_use, met_fix_roll, cm_fixed, Epulse_fix),
        ]

        for label, x_design, met_roll, Cm_val, Epulse_val in design_triplets:
            res = integrate_distribution_energy(
                dist_df,
                Cm_Ns_per_J=Cm_val,
                Epulse_J=Epulse_val,
                eta_abs=eta_abs_roll,
                rho_kg_m3=(rho_bulk_use if 'rho_bulk_use' in locals() else 2700.0),
                hp_km=(hp_km if 'hp_km' in locals() else 300.0),
                xi_geom=(xi_geom if 'xi_geom' in locals() else 1.0),
                count_col="N_total_bin", alt_col="Altitude_km", diam_col="Diameter_m",
            )

            # Energies
            J_to_kWh = 1.0/3.6e6
            optical_kWh    = res["total_optical_J"]  * J_to_kWh
            absorbed_kWh   = res["total_absorbed_J"] * J_to_kWh
            wall_eff       = (eta_wall if 'eta_wall' in locals() else 0.25)
            electrical_kWh = optical_kWh / max(wall_eff, 1e-6)

            # Means 
            n_obj = max(res["total_objects"], 1.0)
            pulses_per_obj_mean = res["total_pulses"] / n_obj
            optical_Wh_per_obj_mean = (optical_kWh * 1e3) / n_obj  # kWh -> Wh

            # ---- COSTS for the full distribution program ----
            area_m2 = math.pi * (x_design['D'] / 2.0)**2
            D = x_design['D']

            # Base pieces
            optics_capex = k_area * area_m2
            P_elec_avg_kW = met_roll['P_elec_avg_W'] / 1e3
            laser_capex   = max(k_laser * P_elec_avg_kW, laser_capex_floor)

            # Nonlinear aperture + fixed program floor
            aperture_capex = cap_ap_coef * (D ** cap_ap_alpha)
            fixed_capex    = program_floor

            # CAPEX with overheads/margins
            capex_total = (fixed_capex + optics_capex + laser_capex + aperture_capex) * overhead_mult

            # OpEx: energy + ops-time
            energy_cost = price_kWh * electrical_kWh
            ops_hours   = res["total_pulses"] / max(x_design['f_rep'], 1e-9) / 3600.0
            ops_cost    = k_time * ops_hours

            total_program_cost = capex_total + energy_cost + ops_cost

            cost_per_object    = total_program_cost / n_obj

            results.append({
                "label": label,
                "C_m (N·s/J)": f"{Cm_val:.0e}",
                "Pulse energy (J)": f"{Epulse_val:,.1f}",
                #"ξ": (xi_geom if 'xi_geom' in locals() else 1.0),
                "Target perigee (km)": (hp_km if 'hp_km' in locals() else 300.0),
                "ρ (kg/m³)": (rho_bulk_use if 'rho_bulk_use' in locals() else 2700.0),

                "Population count": f"{res['total_objects']:,.0f}",
                #"Total pulses": f"{res['total_pulses']:.3e}",

                "Optical energy (kWh)": f"{optical_kWh:,.1f}",
                "Absorbed energy (kWh)": f"{absorbed_kWh:,.1f}",
                "Electrical energy (kWh)": f"{electrical_kWh:,.1f}",

                #"Aperture area (m²)": f"{area_m2:.3f}",
                "Laser P_elec,avg (kW)": f"{P_elec_avg_kW:.2f}",
                "Energy cost ($)": f"{energy_cost:,.0f}",
                "Ops hours": f"{ops_hours:,.1f}",
                "Ops cost ($)": f"{ops_cost:,.0f}",
                "Optics CAPEX ($)": f"{optics_capex:,.0f}",
                "Laser CAPEX ($)": f"{laser_capex:,.0f}",
                "Total program cost ($)": f"{total_program_cost:,.0f}",
                "Cost per object ($/obj)": f"{cost_per_object:,.2f}",

                #"Mean pulses / object": f"{pulses_per_obj_mean:.3e}",
                "Mean optical energy / object (Wh)": f"{optical_Wh_per_obj_mean:,.2f}",
            })

        # Build a two-column summary
        metrics_order = [
            "C_m (N·s/J)",
            "Pulse energy (J)",
            #"ξ",
            "Target perigee (km)",
            "ρ (kg/m³)",
            "Population count",
            "Optical energy (kWh)",
            "Absorbed energy (kWh)",
            "Electrical energy (kWh)",

            "Laser P_elec,avg (kW)",
            "Energy cost ($)",
            "Ops hours",
            "Ops cost ($)",
            "Optics CAPEX ($)",
            "Laser CAPEX ($)",
            "Total program cost ($)",
            "Cost per object ($/obj)",

            "Mean optical energy / object (Wh)",
        ]

        cols = [r["label"] for r in results]
        table = pd.DataFrame(index=metrics_order, columns=cols)
        for r in results:
            for k in metrics_order:
                table.at[k, r["label"]] = r[k]

        st.subheader("Distribution roll-up summary")
        st.dataframe(table, use_container_width=True)

        if do_opt and 'drama_filename' in st.session_state:
            try:
                base_name = st.session_state['drama_filename'].split('.csv')[0]
                lam_nm_str = f"_{int(lam_choice*1e9)}nm"
                rollup_filename = f"{base_name}_distribution_rollup{lam_nm_str}.csv"
                table.to_csv(rollup_filename)
                st.success(f"Saved distribution roll-up to {rollup_filename}")
            except Exception as e:
                st.warning(f"Could not save roll-up CSV: {e}")



# ---------- 1) Δv vs time with tabs for multiple masses (per-Cm design values) ----------
m_list = [0.1, 1.0, 2.5, 5.0, 7.5, 10.0]

# =========================
# Graphs section
# =========================
st.subheader("Figures")


def dv_slope(design, Cm_val, m):
    return (Cm_val * eta_use * design['E_pulse'] * design['f_rep']) / max(m, 1e-30)

left, right = st.columns([6, 1])

with right:
    show_shading = st.checkbox("Shading", value=True, key="shade_all_tabs")

with left:
    tabs = st.tabs([f"m = {m:g} kg" for m in m_list])

    for tab, m_use in zip(tabs, m_list):
        with tab:
            base_slope_min = dv_slope(x_lo, x_lo['cm_min'], m_use)
            base_slope_max = dv_slope(x_hi, x_hi['cm_max'], m_use)
            slope_min = base_slope_min * eta_lo(m_use)
            slope_max = base_slope_max * eta_hi(m_use)

            dv_cap = min(300.0, dv_goal)
            t_end_nom = 1.1 * dv_cap / max(slope_min, 1e-30)  
            t_end = min(t_end_nom, T_eng)                      # cap by engagement window


            # If window came from L-range + v_rel, expand to cover both designs' windows
            # Else (custom time), keep the custom T_eng
            if v_rel_kms is not None:
                T_left  = min(T_eng_lo, T_eng_hi)
                T_right = max(T_eng_lo, T_eng_hi)
                t_needed = 1.1 * dv_cap / max(slope_min, 1e-30)
                t_end    = max(T_right, min(t_needed, 300.0))
            else:
                t_needed = 1.1 * dv_cap / max(slope_min, 1e-30)
                t_end    = min(max(T_eng, 1.0), max(t_needed, T_eng))

            t = np.linspace(0.0, t_end, 400)
            dv_min = slope_min * t
            dv_max = slope_max * t

            figA, axA = plt.subplots(figsize=FIGSIZE_DV)


            axA.plot(t, dv_min, lw=1, label=fr"$C_m={x_lo['cm_min']:.1e}$ (E={x_lo['E_pulse']:.0f} J)")
            axA.plot(t, dv_max, lw=1, label=fr"$C_m={x_hi['cm_max']:.1e}$ (E={x_hi['E_pulse']:.0f} J)")
            axA.axhline(dv_goal, ls="--", color="gray", lw=1, label=fr"Δv target = {dv_goal:g} m/s")

            # True intersections with Δv target
            t_hit_min = dv_goal / max(slope_min, 1e-30)
            t_hit_max = dv_goal / max(slope_max, 1e-30)

            if t_hit_min <= t_end:
                axA.scatter([t_hit_min], [dv_goal], color="C0", zorder=3)
            if t_hit_max <= t_end:
                axA.scatter([t_hit_max], [dv_goal], color="C1", zorder=3)

            # mark the per-design windows as verticals
            if v_rel_kms is not None:
                axA.axvline(T_eng_lo, ls="-.", color="C0", lw=1, alpha=0.6, label=f"T_lo = {T_eng_lo:.1f}s")
                axA.axvline(T_eng_hi, ls="-.", color="C1", lw=1, alpha=0.6, label=f"T_hi = {T_eng_hi:.1f}s")
            else:
                axA.axvline(T_eng, ls="-.", color="k", lw=1, alpha=0.6, label=f"T_eng={T_eng:.1f}s")


            if show_shading:
                y_floor_hi = np.maximum(dv_goal, dv_min)   # above both target and blue curve
                y_floor_lo = np.maximum(dv_goal, dv_min)   # above target and blue curve

                # ORANGE (hi): fill between y_floor_hi and dv_max where x<=T_hi and dv_max>y_floor_hi
                mask_hi = (t <= T_eng_hi) & (dv_max > y_floor_hi)
                if np.any(mask_hi):
                    axA.fill_between(
                        t, y_floor_hi, dv_max,
                        where=mask_hi,
                        alpha=0.20, color="C1",
                        label="Best-case region"
                    )

                # BLUE (lo): between blue and orange, above target, x<=T_lo
                y_lower = y_floor_lo       
                y_upper = dv_max
                mask_lo = (t <= T_eng_lo) & (y_upper > y_lower)
                if np.any(mask_lo):
                    axA.fill_between(
                        t, y_lower, y_upper,
                        where=mask_lo,
                        alpha=0.20, color="C0",
                        label="Conservative region"
                    )

            axA.axvline(T_eng_lo, ls="-.", color="C0", lw=1, alpha=0.6, label="_nolegend_")
            axA.axvline(T_eng_hi, ls="-.", color="C1", lw=1, alpha=0.6, label="_nolegend_")

            handles, labels = axA.get_legend_handles_labels()
            kept = [(h, l) for h, l in zip(handles, labels) if l and l != "_nolegend_"]
            if kept:
                handles, labels = zip(*kept)

            axA.legend(
                handles, labels,
                loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0,
                frameon=True
            )

            figA.subplots_adjust(right=0.78)

            suffix = "" if overspill_model == "None (legacy)" else f" (overspill: {overspill_model})"
            axA.set_title(f"Δv vs time (m = {m_use:g} kg){suffix}")
            axA.set_xlabel("Time (s)")
            axA.set_ylabel("Δv (m/s)")
            axA.grid(True, ls="--", alpha=0.4)

            st.pyplot(figA, use_container_width=False)


st.latex(r"\Delta v(t;m)=\frac{C_m\,\eta_{\mathrm{abs}}\,E_{\mathrm{pulse}}\,f_{\mathrm{rep}}}{m}\,\eta_{\mathrm{geom}}(m)\,t")

# ---------- 2) Δv per pulse vs mass ----------
masses = np.linspace(0.1, 10.0, 300)

E_abs_lo = x_lo['E_pulse'] * eta_use
E_abs_hi = x_hi['E_pulse'] * eta_use

# Δv per pulse curves (include geometry efficiency vs mass)
dv_min_curve = (x_lo['cm_min'] * E_abs_lo) / masses * np.array([eta_lo(m) for m in masses])
dv_max_curve = (x_hi['cm_max'] * E_abs_hi) / masses * np.array([eta_hi(m) for m in masses])

figC, axC = new_fig(figsize=FIGSIZE_BIG)

# Curves
axC.plot(masses, dv_min_curve, lw=1.5,
         label=fr"$C_m={x_lo['cm_min']:.1e}$ (E={x_lo['E_pulse']:.0f} J)")
axC.plot(masses, dv_max_curve, lw=1.5,
         label=fr"$C_m={x_hi['cm_max']:.1e}$ (E={x_hi['E_pulse']:.0f} J)")

# Overspill onsets as vertical guides
if mth_lo is not None:
    axC.axvline(mth_lo, color="C0", ls=":", lw=1, alpha=0.8, label="_nolegend_")
if mth_hi is not None:
    axC.axvline(mth_hi, color="C1", ls=":", lw=1, alpha=0.8, label="_nolegend_")

# Target band shading: only where band intersects between curves
band_lo, band_hi = 0.01, 0.1
y_low  = np.maximum(dv_min_curve, band_lo)
y_high = np.minimum(dv_max_curve, band_hi)
mask   = y_high > y_low

if np.any(mask):
    axC.fill_between(masses[mask], y_low[mask], y_high[mask],
                     alpha=0.25, label="Target 0.01–0.1 m/s")

# Band guide lines (not in legend)
axC.axhline(band_lo, color="gray", ls="--", lw=1, alpha=0.8, label="_nolegend_")
axC.axhline(band_hi, color="gray", ls="--", lw=1, alpha=0.8, label="_nolegend_")

axC.set_yscale("log")
axC.set_ylim(1e-3, 2.0)
axC.set_xlabel("Mass m (kg)")
axC.set_ylabel("Δv per pulse (m/s)")
suffix = "" if overspill_model == "None (legacy)" else f" (overspill: {overspill_model})"
axC.set_title("Δv per pulse vs mass" + suffix)
axC.grid(True, which="both", ls="--", alpha=0.4)

render_with_right_legend(figC, axC)


st.latex(r"\Delta v_{\text{pulse}}(m)=\frac{C_m\,\eta_{\mathrm{abs}}\,E_{\mathrm{pulse}}}{m}\,\eta_{\mathrm{geom}}(m)")

# ---------- 3) Fluence vs spot size ----------
d_cm = np.linspace(2.0, 30.0, 400)

def F_of_d(E_J, dcm, eta=eta_use):
    # Φ = 4 η E / (π d^2), with d in cm, Φ in J/cm² when E in J
    return 4.0 * eta * E_J / (np.pi * dcm**2)

F_curve_lo = F_of_d(x_lo['E_pulse'], d_cm)  # conservative energy
F_curve_hi = F_of_d(x_hi['E_pulse'], d_cm)  # best-case energy

# design spot diameters (m -> cm)
d_lo_cm = d_spot_lo_m * 100.0
d_hi_cm = d_spot_hi_m * 100.0

figB, axB = new_fig(figsize=FIGSIZE_BIG)

# Curves
axB.plot(d_cm, F_curve_lo, lw=1.5,
         label=fr"E = {x_lo['E_pulse']:.0f} J (conservative)")
axB.plot(d_cm, F_curve_hi, lw=1.5,
         label=fr"E = {x_hi['E_pulse']:.0f} J (best)")

# Threshold guide
axB.axhline(req['F_min_J_cm2'], color="gray", ls="--", lw=1,
            label=fr"$\Phi_{{\min}}$ = {req['F_min_J_cm2']:.1f} J/cm²")

# Design spot diameters as vertical guides
axB.axvline(d_lo_cm, color="C0", ls=":", lw=1, alpha=0.85, label="_nolegend_")
axB.axvline(d_hi_cm, color="C1", ls=":", lw=1, alpha=0.85, label="_nolegend_")

# Intersection markers at (d_lo, Φmin) and (d_hi, Φmin)
axB.scatter([d_lo_cm], [req['F_min_J_cm2']], color="C0", zorder=3)
axB.scatter([d_hi_cm], [req['F_min_J_cm2']], color="C1", zorder=3)

axB.set_xlim(d_cm.min(), d_cm.max())
axB.set_ylim(0, 50)
axB.set_xlabel("Spot diameter d (cm)")
axB.set_ylabel("Fluence Φ (J/cm²)")
axB.set_title("Fluence vs spot size")
axB.grid(True, ls="--", alpha=0.4)

render_with_right_legend(figB, axB)


st.latex(r"F(d)=\frac{4\,\eta_{\mathrm{abs}}\,E_{\mathrm{pulse}}}{\pi\,d^2}\quad\text{(with }d\text{ in cm)}")

# ---------- 4) Fluence vs Pulse energy (two optimised spot sizes) ----------
E_range = np.linspace(req['E_min_J'], req['E_max_J'], 400)

# Use optical E_pulse (not absorbed) for this figure:
F_vsE_lo = 4.0 * E_range / (np.pi * d_lo_cm**2)   # Φ(E | d = d_lo_cm)
F_vsE_hi = 4.0 * E_range / (np.pi * d_hi_cm**2)   # Φ(E | d = d_hi_cm)

# Energy needed to hit Φ_min for each spot
E_need_lo = (req['F_min_J_cm2'] * np.pi * d_lo_cm**2) / 4.0
E_need_hi = (req['F_min_J_cm2'] * np.pi * d_hi_cm**2) / 4.0

figD, axD = new_fig(figsize=FIGSIZE_BIG)

# Curves
axD.plot(E_range, F_vsE_lo, lw=1.5, label=fr"d = {d_lo_cm:.2f} cm")
axD.plot(E_range, F_vsE_hi, lw=1.5, label=fr"d = {d_hi_cm:.2f} cm")

# Threshold line
axD.axhline(req['F_min_J_cm2'], color="gray", ls="--", lw=1,
            label=fr"$\Phi_{{\min}}$ = {req['F_min_J_cm2']:.1f} J/cm²")

# Vertical guides at required energies
axD.axvline(E_need_lo, color="C0", ls=":", lw=1, alpha=0.85, label="_nolegend_")
axD.axvline(E_need_hi, color="C1", ls=":", lw=1, alpha=0.85, label="_nolegend_")

# Intersection markers
axD.scatter([E_need_lo], [req['F_min_J_cm2']], color="C0", zorder=3)
axD.scatter([E_need_hi], [req['F_min_J_cm2']], color="C1", zorder=3)

axD.set_xlabel("Pulse energy E (J)")
axD.set_ylabel("Fluence Φ (J/cm²)")
axD.set_title("Fluence vs Pulse energy (two optimised spots)")
axD.grid(True, ls="--", alpha=0.4)

render_with_right_legend(figD, axD)


st.latex(r"F(E)=\frac{4\,E_{\mathrm{pulse}}}{\pi\,d^2}")

# ---------- 5) Achievable Δv in one pass vs mass (overspill-aware) ----------
masses = np.linspace(0.1, 10.0, 300)

# Use per-design engagement windows if they exist (computed earlier alongside the Δv–t plot);
# otherwise fall back to the single T_eng.
T_lo_use = T_eng_lo if 'T_eng_lo' in locals() and T_eng_lo is not None else T_eng
T_hi_use = T_eng_hi if 'T_eng_hi' in locals() and T_eng_hi is not None else T_eng

# Δv achievable in one pass, including geometry efficiency vs mass
dv_per_pass_lo = (x_lo['cm_min'] * eta_use * x_lo['E_pulse'] * x_lo['f_rep'] * T_lo_use) / masses \
                 * np.array([eta_lo(m) for m in masses])
dv_per_pass_hi = (x_hi['cm_max'] * eta_use * x_hi['E_pulse'] * x_hi['f_rep'] * T_hi_use) / masses \
                 * np.array([eta_hi(m) for m in masses])

figP, axP = new_fig(figsize=FIGSIZE_BIG)

# Curves
axP.plot(masses, dv_per_pass_lo, lw=1.5, label=fr"$C_m={x_lo['cm_min']:.1e}$")
axP.plot(masses, dv_per_pass_hi, lw=1.5, label=fr"$C_m={x_hi['cm_max']:.1e}$")

# Target Δv
axP.axhline(dv_goal, color="gray", ls="--", lw=1, label=fr"Δv target = {dv_goal:g} m/s")

# Overspill thresholds as vertical guides
if mth_lo is not None:
    axP.axvline(mth_lo, color="C0", ls=":", lw=1, alpha=0.85, label="_nolegend_")
if mth_hi is not None:
    axP.axvline(mth_hi, color="C1", ls=":", lw=1, alpha=0.85, label="_nolegend_")

# y-limits on log scale ---
axP.set_yscale("log")
y_all = np.concatenate([dv_per_pass_lo, dv_per_pass_hi, np.array([dv_goal])])
y_pos = y_all[y_all > 0]
if y_pos.size:
    y_min = y_pos.min()
    y_max = y_pos.max()
    y_lo_plot = max(1e-3, 10**(math.floor(math.log10(y_min)) - 0.2))
    y_hi_plot =            10**(math.ceil (math.log10(y_max)) + 0.2)
    axP.set_ylim(y_lo_plot, y_hi_plot)

axP.set_xlim(masses.min(), masses.max())

axP.set_xlabel("Mass m (kg)")
axP.set_ylabel("Δv achievable in one pass (m/s)")
suffix = "" if overspill_model == "None (legacy)" else f" (overspill: {overspill_model})"
axP.set_title("Δv in one pass" + suffix)
axP.grid(True, which="major", ls="--", alpha=0.4)
axP.grid(True, which="minor", ls=":",  alpha=0.15)
render_with_right_legend(figP, axP)



# ---------- 6) Total Energy (kWh) vs Mass for C_m bounds (overspill-aware) ----------

# Ensure 'masses' exists (reuse from earlier, or define if missing)
if 'masses' not in locals():
    masses = np.linspace(0.1, 10.0, 300)

# Geometry efficiencies vs mass (clamped to avoid divide-by-zero)
eta_lo_vec = np.maximum(np.array([eta_lo(m) for m in masses]), 1e-12)
eta_hi_vec = np.maximum(np.array([eta_hi(m) for m in masses]), 1e-12)

# From N = (m Δv)/(C_m * η_abs * η_geom * E_pulse) and E_total = N * E_pulse
#  -> E_total = m Δv / (C_m * η_abs * η_geom)
eta_abs_safe = max(eta_use, 1e-12)
E_tot_kWh_minCm = (masses * dv_goal) / (x_lo['cm_min'] * eta_abs_safe * eta_lo_vec) / 3.6e6
E_tot_kWh_maxCm = (masses * dv_goal) / (x_hi['cm_max'] * eta_abs_safe * eta_hi_vec) / 3.6e6

figE, axE = new_fig(figsize=FIGSIZE_BIG)

# Curves
axE.plot(masses, E_tot_kWh_minCm, lw=1.5, label=fr"$C_m = {x_lo['cm_min']:.1e}$")
axE.plot(masses, E_tot_kWh_maxCm, lw=1.5, label=fr"$C_m = {x_hi['cm_max']:.1e}$")

# Overspill thresholds as vertical guides
if mth_lo is not None:
    axE.axvline(mth_lo, color="C0", ls=":", lw=1, alpha=0.85, label="_nolegend_")
if mth_hi is not None:
    axE.axvline(mth_hi, color="C1", ls=":", lw=1, alpha=0.85, label="_nolegend_")

axE.set_xlabel("Mass m (kg)")
axE.set_ylabel("Total energy to deliver (kWh)")
suffix = "" if overspill_model == "None (legacy)" else f" (overspill: {overspill_model})"
axE.set_title("E_total (kWh) vs mass for Δv target" + suffix)
axE.grid(True, ls="--", alpha=0.4)
render_with_right_legend(figE, axE)

st.latex(r"E_{\text{total,optical}}(m)=\frac{m\,\Delta v_{\text{target}}}{C_m\,\eta_{\mathrm{abs}}\,\eta_{\mathrm{geom}}(m)}\ \ \left[\mathrm{J}\right],\quad \text{then divide by }3.6\times10^6\text{ for kWh.}")

# ---- End ----
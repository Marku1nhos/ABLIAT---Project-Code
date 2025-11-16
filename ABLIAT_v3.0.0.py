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

# SciPy optimiser (falls back to random search if missing)
_SCIPY_OK = False
with contextlib.suppress(Exception):
    from scipy.optimize import differential_evolution  # type: ignore
    _SCIPY_OK = True


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

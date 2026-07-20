"""
Zombie Epidemic Simulation Tools
Based on: Munz et al. 2009 "When Zombies Attack! Mathematical Modelling
of an Outbreak of Zombie Infection"

Tools:
    run_zombie_simulation  – SZR / SIZR ODE solver (pure Python for-loop)
    run_monte_carlo        – Monte Carlo survival analysis with NumPy
    find_optimal_attack    – Binary search for minimum kill-rate (alpha)
"""

import numpy as np


# ==========================================
# TOOL 1: run_zombie_simulation
# ==========================================

def run_zombie_simulation(
    alpha: float,
    beta: float,
    zeta: float,
    delta: float,
    model: str = "SZR",
) -> str:
    """
    Runs a zombie epidemic simulation from Munz et al. 2009.

    Args:
        alpha: Zombie kill rate (0.001 – 0.015)
        beta:  Infection rate   (0.005 – 0.020)
        zeta:  Resurrection rate (0.00001 – 0.001)
        delta: Natural death rate (0.00001 – 0.001)
        model: "SZR" (basic) or "SIZR" (with latency class, rho=0.005)

    Returns:
        Formatted string with final S, Z, R values, the day zombies first
        outnumbered humans, and a survival verdict.
    """
    N   = 500
    S0  = 499
    Z0  = 1
    R0  = 0
    dt  = 0.001
    T   = 10            # simulation days
    steps = int(T / dt)
    rho   = 0.005       # latency-to-zombie rate (SIZR only)

    S = float(S0)
    Z = float(Z0)
    R = float(R0)
    I = 0.0             # Infected (latent) class – used only in SIZR

    overtake_day = None

    for step in range(steps):
        t = step * dt

        if model == "SIZR":
            dS = -beta * S * Z - delta * S
            dI =  beta * S * Z - rho * I - delta * I
            dZ =  zeta * R + rho * I - alpha * S * Z
            dR =  delta * S + delta * I + alpha * S * Z - zeta * R

            S = max(S + dS * dt, 0.0)
            I = max(I + dI * dt, 0.0)
            Z = max(Z + dZ * dt, 0.0)
            R = max(R + dR * dt, 0.0)
        else:  # SZR (basic)
            dS = -beta * S * Z - delta * S
            dZ =  beta * S * Z + zeta * R - alpha * S * Z
            dR =  delta * S + alpha * S * Z - zeta * R

            S = max(S + dS * dt, 0.0)
            Z = max(Z + dZ * dt, 0.0)
            R = max(R + dR * dt, 0.0)

        # Record first day zombies outnumber humans
        if overtake_day is None and Z > S:
            overtake_day = round(t, 3)

    S_final = round(S, 2)
    Z_final = round(Z, 2)
    R_final = round(R, 2)
    verdict = "[SURVIVE] HUMANS SURVIVE" if S_final > 50 else "[ZOMBIES WIN]"

    overtake_str = (
        f"Day {overtake_day}" if overtake_day is not None else "Never (humans held the line)"
    )

    return (
        f"=== Zombie Simulation ({model} model) ===\n"
        f"Parameters : alpha={alpha}, beta={beta}, zeta={zeta}, delta={delta}\n"
        f"Final State (Day {T}):\n"
        f"  Susceptible (S) : {S_final}\n"
        f"  Zombies     (Z) : {Z_final}\n"
        f"  Removed     (R) : {R_final}\n"
        f"Zombies overtook humans : {overtake_str}\n"
        f"Verdict : {verdict}\n"
    )


# ==========================================
# TOOL 2: run_monte_carlo
# ==========================================

def run_monte_carlo(n_scenarios: int = 1000) -> str:
    """
    Runs Monte Carlo analysis across multiple random parameter scenarios.

    Args:
        n_scenarios: Number of random scenarios to simulate (max 5000).

    Returns:
        Survival probability, mean survivors, and the alpha range that
        gives the best survival outcomes.
    """
    # pyrefly: ignore [unnecessary-type-conversion]
    n_scenarios = min(int(n_scenarios), 5000)

    rng = np.random.default_rng(seed=42)

    beta_samples  = rng.uniform(0.005, 0.015, n_scenarios)
    alpha_samples = rng.uniform(0.001, 0.015, n_scenarios)
    zeta_samples  = rng.uniform(0.00001, 0.001, n_scenarios)

    # Fixed simulation params for MC runs (shorter horizon: 5 days)
    delta = 0.0001
    dt    = 0.001
    T     = 5
    steps = int(T / dt)

    S_finals    = []
    survived    = 0
    best_alpha  = None
    best_S      = -1.0

    for i in range(n_scenarios):
        beta  = beta_samples[i]
        alpha = alpha_samples[i]
        zeta  = zeta_samples[i]

        S = 499.0
        Z = 1.0
        R = 0.0

        for _ in range(steps):
            dS = -beta * S * Z - delta * S
            dZ =  beta * S * Z + zeta * R - alpha * S * Z
            dR =  delta * S + alpha * S * Z - zeta * R
            S  = max(S + dS * dt, 0.0)
            Z  = max(Z + dZ * dt, 0.0)
            R  = max(R + dR * dt, 0.0)

        S_f = round(S, 2)
        S_finals.append(S_f)

        if S_f > 50:
            survived += 1

        if S_f > best_S:
            best_S     = S_f
            best_alpha = alpha

    survival_prob = survived / n_scenarios * 100
    mean_S        = round(float(np.mean(S_finals)), 2)

    return (
        f"=== Monte Carlo Analysis ({n_scenarios} scenarios) ===\n"
        f"Survival Probability      : {survival_prob:.1f}%\n"
        f"Mean Survivors (S_final)  : {mean_S}\n"
        f"Best Alpha Found          : {best_alpha:.5f}  => {best_S:.1f} survivors\n"
        f"Parameter Ranges Sampled  :\n"
        f"  beta  ~ U(0.005, 0.015)\n"
        f"  alpha ~ U(0.001, 0.015)\n"
        f"  zeta  ~ U(0.00001, 0.001)\n"
        f"Note: Munz 2009 baseline (alpha=0.005) rarely produces survivors.\n"
    )


# ==========================================
# TOOL 3: find_optimal_attack
# ==========================================

def find_optimal_attack(target_survivors: int = 100) -> str:
    """
    Finds the minimum kill rate (alpha) needed for humans to survive.

    Args:
        target_survivors: Minimum number of surviving humans required.

    Returns:
        Minimum alpha needed, recommended strategy, and comparison with
        the Munz 2009 baseline.
    """
    # Fixed parameters (Munz 2009 baseline for everything except alpha)
    beta  = 0.0095
    zeta  = 0.0001
    delta = 0.0001
    dt    = 0.001
    T     = 10
    steps = int(T / dt)

    def simulate_S_final(alpha: float) -> float:
        S = 499.0
        Z = 1.0
        R = 0.0
        for _ in range(steps):
            dS = -beta * S * Z - delta * S
            dZ =  beta * S * Z + zeta * R - alpha * S * Z
            dR =  delta * S + alpha * S * Z - zeta * R
            S  = max(S + dS * dt, 0.0)
            Z  = max(Z + dZ * dt, 0.0)
            R  = max(R + dR * dt, 0.0)
        return S

    # Binary search over alpha ∈ [0.001, 0.020]
    lo, hi    = 0.001, 0.020
    tolerance = 1e-6
    min_alpha = None

    for _ in range(60):          # 60 iterations → precision ~1e-6
        mid = (lo + hi) / 2
        S_f = simulate_S_final(mid)
        if S_f >= target_survivors:
            min_alpha = mid
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tolerance:
            break

    # Munz 2009 baseline result for comparison
    baseline_S = round(simulate_S_final(0.005), 2)

    if min_alpha is None:
        return (
            f"=== Optimal Attack Search ===\n"
            f"Target Survivors : {target_survivors}\n"
            f"Result : ❌ No alpha in [0.001, 0.020] achieves the target.\n"
            f"Munz 2009 baseline (alpha=0.005) → S_final = {baseline_S}\n"
            f"Recommendation : Seek additional interventions (quarantine, cure research).\n"
        )

    min_alpha_rounded = round(min_alpha, 6)
    achieved_S        = round(simulate_S_final(min_alpha), 2)

    # Provide strategic interpretation
    if min_alpha_rounded <= 0.005:
        strategy = "Standard military response (Munz 2009 baseline) is sufficient."
    elif min_alpha_rounded <= 0.010:
        strategy = "Enhanced strike operations required (2x Munz baseline)."
    else:
        strategy = "Maximum-effort eradication campaign needed. Consider SIZR containment."

    return (
        f"=== Optimal Attack Search ===\n"
        f"Target Survivors        : {target_survivors}\n"
        f"Minimum Alpha Required  : {min_alpha_rounded}\n"
        f"Survivors Achieved      : {achieved_S}\n"
        f"Munz 2009 Baseline      : alpha=0.005 => S_final={baseline_S}\n"
        f"Improvement Needed      : {round(min_alpha_rounded / 0.005, 2)}x baseline kill-rate\n"
        f"Recommended Strategy    : {strategy}\n"
    )

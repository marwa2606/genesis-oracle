def get_system_state(kappa: float) -> tuple[str, str]:
    """
    Returns the system state and a temperature log based on the current kappa value.
    - FREEZING if kappa < 0.4
    - PERFECT if 0.4 <= kappa <= 0.8
    - BOILING if kappa > 0.8
    """
    if kappa < 0.4:
        state = "FREEZING"
    elif kappa <= 0.8:
        state = "PERFECT"
    else:
        state = "BOILING"
        
    log_msg = f"Thermal sensor log: Kappa coefficient is at {kappa:.4f}. Current system state categorized as {state}."
    return state, log_msg

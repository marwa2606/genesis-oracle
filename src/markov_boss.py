import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def simulate_agent(key, P_seq):
    """
    Simulates the state transitions of a single agent over 365 steps.
    All agents start in State 0 (Bull Market).
    """
    def step_fn(carry, P_t):
        state, k = carry
        k, subk = jax.random.split(k)
        # Select the transition probability row for the current state
        probs = P_t[state]
        # Transition to the next state using JAX random choice
        next_state = jax.random.choice(subk, 3, p=probs)
        return (next_state, k), next_state

    # Initial state is 0, scan over the sequence of transition matrices
    _, states = jax.lax.scan(step_fn, (0, key), P_seq)
    return states

def main():
    print("Initializing Markov Chain Simulation with pure JAX...")
    
    # 1. Define baseline transition matrix P
    P_base = jnp.array([
        [0.85, 0.12, 0.03],
        [0.10, 0.75, 0.15],
        [0.05, 0.20, 0.75]
    ])
    
    # 3. Define Black Swan Shock transition matrix (Days 180-190)
    P_shock = jnp.array([
        [0.05, 0.15, 0.80],
        [0.05, 0.15, 0.80],
        [0.05, 0.20, 0.75]
    ])
    
    # Generate the sequence of transition matrices for 365 steps
    # Day 0 is the starting state, step 0 to 364 represent the 365 transitions.
    P_seq = jnp.repeat(P_base[None, :, :], 365, axis=0)
    # Apply shock on days 180-190 (inclusive, which is indices 180 to 190)
    P_seq = P_seq.at[180:191].set(P_shock)
    
    # 2. Simulate 100,000 agents in parallel using vmap
    num_agents = 100_000
    master_key = jax.random.PRNGKey(42)
    keys = jax.random.split(master_key, num_agents)
    
    print(f"Simulating {num_agents:,} agents over 365 days...")
    vmapped_simulator = jax.jit(jax.vmap(lambda k: simulate_agent(k, P_seq)))
    transitioned_states = vmapped_simulator(keys)
    
    # Prepend day 0 initial state (all agents in State 0)
    initial_states = jnp.zeros((num_agents, 1), dtype=jnp.int32)
    all_states = jnp.concatenate([initial_states, transitioned_states], axis=1)
    
    # Calculate state distribution percentages for each day (0 to 365)
    prop_bull = jnp.mean(all_states == 0, axis=0) * 100.0
    prop_stag = jnp.mean(all_states == 1, axis=0) * 100.0
    prop_rece = jnp.mean(all_states == 2, axis=0) * 100.0
    
    # 5. Print final state distribution at day 365
    print("\n" + "=" * 45)
    print("        FINAL STATE DISTRIBUTION (DAY 365)        ")
    print("=" * 45)
    print(f"State 0 (Bull Market):  {prop_bull[365]:.2f}%")
    print(f"State 1 (Stagnation):   {prop_stag[365]:.2f}%")
    print(f"State 2 (Recession):    {prop_rece[365]:.2f}%")
    print("=" * 45 + "\n")
    
    # 6. Print 3-sentence summary of revenue model collapse
    print("REVENUE COLLAPSE SUMMARY:")
    summary_text = (
        "During the recession shock spike (Days 180-190), the probability of entering a recession reaches 80% for "
        "almost all states, driving over 76% of agents into a recession simultaneously. This massive transition would "
        "cause a catastrophic drop in market demand (D) while simultaneously skyrocketing the log-normal cost volatility "
        "(sigma), mimicking the stress-test failure where VaR_95% falls below zero. Consequently, the revenue model "
        "from Exercise 2 would experience negative cash flows and a complete loss of viability due to the co-occurrence of "
        "depressed sales and extreme downside cost spikes."
    )
    print(summary_text)
    print("=" * 45 + "\n")
    
    # 4. Track and plot the percentage of agents in each state
    print("Generating premium state transition plot...")
    os.makedirs("data", exist_ok=True)
    
    plt.figure(figsize=(12, 7), dpi=300)
    
    days = jnp.arange(366)
    
    # Plot proportions with curated, harmonious premium HSL-like colors
    plt.plot(days, prop_bull, color="#10b981", linewidth=2.5, label="State 0 (Bull Market)")
    plt.plot(days, prop_stag, color="#f59e0b", linewidth=2.5, label="State 1 (Stagnation)")
    plt.plot(days, prop_rece, color="#ef4444", linewidth=2.5, label="State 2 (Recession)")
    
    # Mark Day 180 and 190 with vertical dashed lines
    plt.axvline(180, color="#64748b", linestyle="--", linewidth=1.5, label="Shock Starts (Day 180)")
    plt.axvline(190, color="#475569", linestyle="-.", linewidth=1.5, label="Shock Ends (Day 190)")
    
    # Highlight the shock region
    plt.axvspan(180, 190, color="#ef4444", alpha=0.1, label="Black Swan Shock Window")
    
    # Add title and labels with premium typography and padding
    plt.title("Markov Chain Multi-Agent Simulation: Black Swan Shock Impact", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Simulation Day", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Percentage of Agents (%)", fontsize=11, fontweight="bold", labelpad=10)
    
    # Styling details
    plt.xlim(0, 365)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    plt.gca().set_facecolor("#f8fafc")
    plt.gcf().patch.set_facecolor("white")
    
    # Premium legend placement
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0", shadow=True, fontsize=10)
    
    # Save image
    plot_path = "data/markov_states.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"State transition plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()

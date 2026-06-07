import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def simulate_path(key):
    """
    Simulates a single business scenario path using pure JAX.
    
    1. Splits the key into 3 subkeys.
    2. Samples Demand D ~ Normal(mu=1000, sigma=150) using jax.random.normal.
    3. Samples Cost C ~ LogNormal(mu=5.5, sigma=0.3) using jnp.exp(jax.random.normal).
    4. Samples Rate R ~ Uniform(0.05, 0.25) using jax.random.uniform.
    5. Returns Revenue = (D * 150.0) - C * (1.0 - R).
    """
    # 1. Split key into 3 subkeys
    key_d, key_c, key_r = jax.random.split(key, 3)
    
    # 2. Sample D ~ Normal(mu=1000, sigma=150)
    # jax.random.normal returns standard normal (mu=0, sigma=1)
    D = 1000.0 + 150.0 * jax.random.normal(key_d)
    
    # 3. Sample C ~ LogNormal(mu=5.5, sigma=0.3)
    # LogNormal is modeled as exp(mu + sigma * normal_sample)
    C = jnp.exp(5.5 + 0.3 * jax.random.normal(key_c))
    
    # 4. Sample R ~ Uniform(0.05, 0.25)
    R = jax.random.uniform(key_r, minval=0.05, maxval=0.25)
    
    # 5. Return Revenue
    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue

def main():
    print("Initializing JAX Monte Carlo business simulation...")
    
    # Master key: jax.random.PRNGKey(42)
    master_key = jax.random.PRNGKey(42)
    
    # Generate 1,000,000 subkeys with jax.random.split (NO for-loop!)
    num_simulations = 1_000_000
    print(f"Generating {num_simulations:,} subkeys in parallel...")
    keys = jax.random.split(master_key, num_simulations)
    
    # Apply jax.vmap(simulate_path) over all subkeys
    # We compile the function with jax.jit for high-performance JAX execution.
    print("Running simulations using JAX vmap and JIT compilation...")
    vmapped_simulator = jax.jit(jax.vmap(simulate_path))
    revenues = vmapped_simulator(keys)
    
    # Ensure all asynchronous JAX execution has finished to get accurate measurements/timing
    revenues.block_until_ready()
    
    # Calculate and print Expected Revenue and VaR_95%
    expected_revenue = float(jnp.mean(revenues))
    var_95 = float(jnp.percentile(revenues, 5.0))
    
    print("\n" + "=" * 45)
    print("            SIMULATION RESULTS            ")
    print("=" * 45)
    print(f"Expected Revenue (mean):   ${expected_revenue:,.2f}")
    print(f"VaR_95% (5th percentile):  ${var_95:,.2f}")
    print("=" * 45 + "\n")
    
    # Ensure the output directory 'data' exists
    os.makedirs("data", exist_ok=True)
    
    # Generate a histogram plot
    print("Creating publication-quality distribution plot...")
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot histogram with aesthetic styling
    plt.hist(
        revenues,
        bins=100,
        color="#2b5c8f",
        edgecolor="#1f4268",
        alpha=0.75,
        density=True,
        label="Revenue Distribution Scenarios"
    )
    
    # Draw Expected Revenue line (solid black)
    plt.axvline(
        expected_revenue,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label=f"Expected Revenue: ${expected_revenue:,.2f}"
    )
    
    # Draw VaR_95% line (dashed red)
    plt.axvline(
        var_95,
        color="#d9534f",
        linestyle="--",
        linewidth=2.5,
        label=f"VaR 95% (5th Percentile): ${var_95:,.2f}"
    )
    
    # Plot styling
    plt.title("Monte Carlo Business Simulation - Revenue Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Revenue ($)", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold", labelpad=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Format x-axis as currency
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    # Legend
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    
    # Save the plot
    plot_path = "data/revenue_dist.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Histogram successfully generated and saved to: {plot_path}")

if __name__ == "__main__":
    main()

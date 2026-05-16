import time
import jax
import jax.numpy as jnp

# Simulation parameters
dt = 0.01
gamma = 0.1
num_oscillators = 100000
num_steps = 1000

def oscillator_step(x, v, w):
    """
    Pure JAX function implementing a single integration step for a 
    damped harmonic oscillator using explicit Euler method.
    """
    v_new = v - dt * (w**2 * x + 2 * gamma * v)
    x_new = x + dt * v
    return x_new, v_new

# We use jax.vmap to vectorize the `oscillator_step` function over the `w` 
# (frequency) array, as well as the states `x` and `v`. 
# This completely eliminates the need for a Python for-loop over the 100,000 
# frequencies, allowing JAX to execute the operations in parallel as batched arrays.
vmap_oscillator_step = jax.vmap(oscillator_step)

@jax.jit
def run_simulation(x, v, w):
    """
    Main simulation loop iterating over time steps.
    By wrapping the outer simulation loop with @jax.jit, we trigger Just-In-Time
    compilation. JAX traces the entire loop and compiles it into a single, highly
    optimized XLA (Accelerated Linear Algebra) operation. This avoids Python 
    interpreter overhead during the loop execution and vastly improves performance.
    """
    # Using jax.lax.scan is the optimal way to write compiled loops in JAX, 
    # preventing long compilation times that can happen with unrolled Python for-loops.
    def step_fn(carry, _):
        curr_x, curr_v = carry
        next_x, next_v = vmap_oscillator_step(curr_x, curr_v, w)
        return (next_x, next_v), None
    
    (final_x, final_v), _ = jax.lax.scan(step_fn, (x, v), None, length=num_steps)
    return final_x, final_v

def main():
    # Initialize random frequencies
    key = jax.random.PRNGKey(42)
    w = jax.random.uniform(key, shape=(num_oscillators,), minval=0.5, maxval=2.0)
    
    # Initialize state
    x = jnp.ones(num_oscillators)
    v = jnp.zeros(num_oscillators)
    
    print("Starting first run (warm-up/compilation)...")
    # First run: warm-up (not timed) - JAX compiles the function on the first call
    x_final, v_final = run_simulation(x, v, w)
    
    # JAX uses asynchronous dispatch. We must block_until_ready() to ensure
    # the computation is actually complete before proceeding.
    x_final.block_until_ready()
    print("Warm-up complete.\n")
    
    print("Starting second run (measured)...")
    # Second run: measure with time.perf_counter()
    start_time = time.perf_counter()
    x_final, v_final = run_simulation(x, v, w)
    x_final.block_until_ready() # Wait for completion to measure accurate time
    end_time = time.perf_counter()
    
    second_run_time = end_time - start_time
    
    print(f"JAX Simulation Time: {second_run_time:.4f} seconds")
    
    # Legacy comparison
    legacy_time = 1.8924
    speedup = legacy_time / second_run_time
    print(f"Speedup Factor = {legacy_time} / {second_run_time:.4f} = {speedup:.2f}x")

if __name__ == "__main__":
    main()

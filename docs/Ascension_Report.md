# Ascension Report – Problem Set 4

## Exercise 2: Speedup Factor

- **Legacy Time (NumPy):** 1.8924 seconds
- **JAX 2nd Run Time:** 0.0078 seconds  
- **Speedup Factor:** 241.65x

### Why is the first JIT run always slower?
The first JIT run is always slower because it triggers the XLA compilation (tracing phase), where JAX converts your Python code into a highly optimized computation graph tailored for the specific hardware. During the second run, JAX skips this tracing overhead entirely and directly executes the already-compiled binary, resulting in dramatically faster performance.

## Exercise 3: Time Travel via Gradients (jax.grad)

### Optimization Results
- **Starting v_initial:** 10.0 m/s
- **Target distance:** 150.0 meters
- **Iterations:** 200
- **Learning rate:** 0.001
- **Final optimized v_initial:** 38.3601 m/s
- **Final Loss:** 0.0000 (converged)

The optimizer converged at iteration ~100, confirming successful gradient descent.

### jax.grad vs Finite Differences
`jax.grad` computes the exact analytical gradient by applying the chain rule automatically through the computation graph (Automatic Differentiation). In contrast, finite differences approximate the gradient by evaluating f(x+h) and f(x) separately, which introduces numerical errors and requires at least 2 function evaluations per parameter. Because `jax.grad` is mathematically exact, it is faster and scales to millions of parameters without additional cost, while finite differences become prohibitively expensive in high dimensions.

## Exercise 4: Agentic Refactoring for the Horizon (Flax)

### Execution Results
- **Output Shape:** (4, 1)
- **Sample Output:**
  [[-0.15686697], [ 0.41225025], [ 0.9883276 ], [ 0.75024045]]

### Keras vs Flax: Explicit State Management
In Keras, model weights are stored inside the model object itself (model.weights),
creating implicit hidden state that violates JAX's functional purity rules.
In Flax, model.init(jax.random.PRNGKey(0), x) returns a separate params dictionary
that lives completely outside the model object.
The forward pass model.apply(params, x) receives the parameters explicitly as an
argument, making the model a pure stateless function.
This separation makes the architecture fully compatible with jax.jit, jax.grad,
and jax.vmap since there is no hidden mutable state.

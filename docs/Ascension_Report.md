# Ascension Report – Problem Set 4

## Exercise 2: Speedup Factor

- **Legacy Time (NumPy):** 1.8924 seconds
- **JAX 2nd Run Time:** 0.0078 seconds  
- **Speedup Factor:** 241.65x

### Why is the first JIT run always slower?
The first JIT run is always slower because it triggers the XLA compilation (tracing phase), where JAX converts your Python code into a highly optimized computation graph tailored for the specific hardware. During the second run, JAX skips this tracing overhead entirely and directly executes the already-compiled binary, resulting in dramatically faster performance.

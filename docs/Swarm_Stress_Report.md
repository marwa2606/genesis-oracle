# Swarm Stress Report

This report summarizes the findings of the automated task sweep evaluating our `src/monte_carlo.py` pipeline.

## Task 1 - Subagent-Alpha (The Stress-Tester)

### Objective
Modify the `sigma` ($\sigma$) parameter of the Log-Normal asset cost distribution ($C$) and evaluate the impact on Expected Revenue and Value at Risk (VaR_95%). Find the breaking point where VaR_95% drops below zero.

### Results
| Sigma ($\sigma$) | Expected Revenue ($) | VaR_95% ($) | Status | Notes |
|-------------------|---------------------:|------------:|:------:|:------|
| 0.3 | $149,751.72 | $112,734.47 | PASS | Requested range |
| 0.5 | $149,733.64 | $112,715.03 | PASS | Requested range |
| 0.8 | $149,683.14 | $112,659.37 | PASS | Requested range |
| 1.0 | $149,627.02 | $112,595.98 | PASS | Requested range |
| 1.5 | $149,332.66 | $112,188.73 | PASS | Requested range |
| 2.0 | $148,456.77 | $110,624.70 | PASS | Requested range |
| 2.5 | $145,409.95 | $106,766.91 | PASS | Extended range |
| 3.0 | $132,772.56 | $98,868.56 | PASS | Extended range |
| 3.5 | $70,719.18 | $77,027.99 | PASS | Extended range |
| 4.0 | $-282,180.97 | $-333.83 | FAIL | Extended range |
| 4.5 | $-2,542,197.75 | $-186,402.12 | FAIL | Extended range |

### Breaking Point Analysis
- **Breaking Point ($\sigma$):** `4.0`
- **Reasoning:** In the requested testing range ($0.3 \le \sigma \le 2.0$), the VaR_95% remained positive, dropping only to **$110,624.70** for $\sigma = 2.0$. By extending the parameter search, we identified the true breaking point at **\sigma = 4.0**, where the 5th percentile of revenue falls to **$-333.83**. This negative VaR indicates that extreme cost volatility can lead to net losses.

---

## Task 2 - Subagent-Beta (The Profiler)

### Objective
Execute the simulation twice sequentially to measure wall-clock execution time and calculate the JIT compilation overhead.

### Subprocess Execution Profiling (Sequential Runs)
- **Run 1 Wall-Clock Time:** 4.5995 seconds
- **Run 2 Wall-Clock Time:** 4.4035 seconds
- **Subprocess Difference (Run1 - Run2):** 0.1960 seconds

> [!NOTE]
> When executing the script as separate subprocesses, both runs start a fresh Python interpreter and must compile the JAX functions from scratch. In addition, the startup and module import times of large packages (`jax`, `matplotlib`) dominate the runtime, masking the JIT compilation speedup.

### In-Process JIT Compilation Profiling
To isolate the actual JIT compilation overhead from interpreter startup and library import times, we measured sequential invocations of the JIT-compiled function within the same Python process:

- **Cold Run (Compilation + Execution):** 2.0804 seconds
- **Warm Run (Execution Only):** 0.4542 seconds
- **Isolated JIT Compilation Overhead:** 1.6262 seconds

### Summary of Profiling Findings
The compilation step takes approximately **1.6262 seconds** (around **78.2%** of the initial call duration). Subsequent executions run **4.6x faster**, highlighting the significant performance advantages of JAX's JIT compilation model once the optimized machine code graph is cached in memory.

# Agent Report

**Status:** Successful Execution Confirmed

I have successfully executed the `src/ancients.py` simulation script. I can confirm that the resulting plot image was successfully created and saved in the `data` folder as `ode_solutions.png`.

## Simulated Physical Systems

The script utilizes `scipy.integrate.solve_ivp` to model and simulate two distinct differential equation systems:

1. **Linearized Pendulum (Simple Harmonic Oscillator):**
   - **Equation:** $x'' + \omega^2 x = 0$
   - **Description:** Simulates the continuous, stable sinusoidal oscillation of a pendulum with an angular frequency $\omega = 2.0$. It tracks both the position $x(t)$ and velocity $x'(t)$ over time.

2. **Radioactive Decay (Exponential Decay):**
   - **Equation:** $x' = -\alpha x$
   - **Description:** Models the first-order exponential decay of a radioactive substance with a decay constant $\alpha = 0.5$. It tracks the remaining quantity $x(t)$ as it approaches zero.

Both systems were evaluated accurately, and the numerical solutions were plotted side-by-side in the output image.

## Anomaly Detection with Conv1D Autoencoder

We have successfully implemented and trained a 1D Convolutional Autoencoder to detect anomalies in a simulated physical signal. The signal contains a Fourier square wave with an RC low-pass filter, Gaussian noise, and an injected high-frequency voltage spike. 

The autoencoder learns to reconstruct the normal periods of the signal. When predicting on the test set, the known anomaly region exhibits a significantly higher Mean Squared Error (MSE) in reconstruction, successfully isolating the injected spike.

![Anomaly Detection](anomaly_detection.png)

## Week 5: Physics-Informed Neural Network (PINN) for the 1D Heat Equation

This week we built a complete Physics-Informed Neural Network pipeline in pure JAX/Flax to solve the 1D Heat Equation on a metal rod (domain: $x \in [-1, 1]$, $t \in [0, 1]$, thermal diffusivity $\alpha = 0.05$).

### Dataset Generation (`src/pinn_data.py`)
Generated three JAX datasets with explicit PRNGKey management:
- **Collocation Points (PDE):** 5,000 random $(x, t)$ interior points for enforcing the PDE residual.
- **Initial Condition (IC):** 500 points at $t=0$ with $u_0(x) = -\sin(\pi x)$.
- **Boundary Conditions (BC):** 500 points (250 at $x=-1$, 250 at $x=+1$) with $u=0$ (Dirichlet).

### Model & Training (`src/fabric_pinn.py`)
- **Architecture:** `HeatSurrogate` MLP — 4 hidden layers × 32 neurons, `tanh` activation (required for smooth second-order derivatives via `jax.grad`).
- **Physics Loss:** PDE residual $(\partial_t u - \alpha \, \partial_{xx} u)^2$ computed with nested `jax.grad` and vectorized over all collocation points with `jax.vmap`.
- **Training:** 5,000 epochs with Optax Adam (lr=1e-3) and a `@jax.jit`-compiled training step. Final loss: **0.000042**.
- **Visualization:** Predictions on a 100×100 meshgrid rendered as an interactive 3D Plotly surface plot.

[View Interactive Plot](pinn_3d_fabric.html)

### The Operator Horizon: Fourier Neural Operators (FNOs)
We also explored how FNOs overcome the PINN's limitation of being tied to a single initial condition:
- **Functional Space Mapping:** FNOs learn an operator from the entire input function space to the solution space, rather than a point-wise coordinate mapping.
- **Frequency Domain Convolutions:** Global integral operators via FFT capture long-range spatial dependencies more efficiently than local grid convolutions.
- **Zero-Shot Generalization:** Weights parameterize the continuous frequency domain, enabling instant predictions for entirely new initial conditions without retraining.

Full details in the [Fabric Report](docs/Fabric_Report.md).

## Week 6: High-Performance Simulations (Monte Carlo & Markov Chain)

This week, we focused on implementing high-performance numerical simulations utilizing NumPy and pure JAX for parallelized execution and profiling.

### Classical Pi Estimation ([src/classical_pi.py](../src/classical_pi.py))
- **Method:** Implemented a standard Monte Carlo Pi estimation utilizing NumPy's uniform distribution to sample points inside a unit square.
- **Visualization:** Outputs a color-coded scatter plot of inside/outside points saved to `data/classical_pi_disp.png`.

### High-Performance JAX Monte Carlo ([src/monte_carlo.py](../src/monte_carlo.py))
- **Parallelization & JIT:** Re-implemented the business revenue model using `jax.vmap` for parallel scenario simulation (1,000,000 runs) and `jax.jit` for compiling the computational graph via XLA.
- **Stress-Testing & Profiling:** Conducted parameter sweep testing on cost volatility ($\sigma$) and analyzed JAX JIT compilation overhead. The complete analysis is documented in the [Swarm Stress Report](Swarm_Stress_Report.md).

### Markov Chain Economic Simulation ([src/markov_boss.py](../src/markov_boss.py))
- **Multi-Agent Simulation:** Modeled 100,000 independent agents transitioning across three economic states (Bull Market, Stagnation, Recession) over 365 days using JAX `lax.scan` and `vmap`.
- **Black Swan Shock:** Injected an 11-day recession shock (Days 180-190) where recession probability spiked to 80%, driving over 76% of agents into a recession simultaneously.
- **Visualization:** Tracks state proportions daily and outputs a premium time-series visualization to `data/markov_states.png`.

## Week 7: Oracle Integration & AI Security

This week, we integrated the system with Google's Gemini API (using the new `google-genai` SDK) to build interactive visual auditing tools, structured decision-making loops, and defensive prompt security configurations.

### Oracle Connectivity ([src/oracle_ping.py](../src/oracle_ping.py))
- Established API connectivity with the `gemini-2.5-flash` model. 
- Integrated custom `GEMINI_API_KEY` environment loading to ensure no keys are hardcoded in the codebase.

### Multimodal Visual Audit ([src/generate_signals.py](../src/generate_signals.py) & [src/visual_audit.py](../src/visual_audit.py))
- **Signal Generation:** Generated a 500-step sine wave and injected a random 20-step clipping/saturation artifact. Plotted the results in dark mode and saved as `data/audit_target.png`.
- **AI Investigation:** Used Gemini's multimodal capacity to read the plot bytes directly, locate the X-axis bounds of the anomaly, and compose a poem mocking the team for allowing the bug to pass.

### Structured Control Loops ([src/sandbox_env.py](../src/sandbox_env.py) & [src/game_loop.py](../src/game_loop.py))
- Implemented a 5-step thermal dampener controller where `gemini-2.5-flash` receives telemetry logs and makes control adjustments.
- Structured Gemini's outputs into a type-safe Pydantic schema (`ControlDecision`) using structured JSON mode to guarantee valid state changes.

### Prompt Hardening and Security ([src/defensive_agent.py](../src/defensive_agent.py))
- Built a secure log parser comparing unprotected prompts against hardened variants.
- Demonstrated defenses against prompt injection attacks by leveraging XML delimiters, strict system role definitions, negative constraints, and explicit anomaly flagging.

Full details are documented in the [Cerebral Nexus Report](Cerebral_Nexus_Report.md).



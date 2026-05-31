## Fabric Report - Problem Set 5

### Exercise 4: Interactive 3D Visualization
[Screenshot placeholder]
[HTML link placeholder]

### Exercise 5: The Operator Horizon (FNOs)
- **Mapping Functional Spaces:** While a PINN learns a mapping from specific spatial-temporal coordinates $(x,t)$ to a scalar $u$ (requiring complete retraining for any new initial conditions), an FNO learns an operator mapping an entire input functional space (e.g., the initial condition $u_0(x)$) directly to the continuous solution functional space $u(x,t)$. 
- **Frequency Domain Convolutions:** FNOs compute global integral operators using Fast Fourier Transforms (FFT). By filtering and applying linear weight transformations only to the lower-frequency modes (which dominate the macroscopic physics of PDEs), the network captures global spatial dependencies far more efficiently than standard, localized grid convolutions.
- **Zero-Shot Generalization (Resolution-Invariance):** Because FNO weights parameterize the continuous frequency domain rather than discrete spatial grids, the model learns the underlying abstract mathematical operator. Once trained, it is completely resolution-invariant and can perform instant, zero-shot predictions for entirely new initial condition functions without ever needing retraining.

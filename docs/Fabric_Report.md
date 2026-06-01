## Fabric Report - Problem Set 5

### Exercise 4: Interactive 3D Visualization
The PINN was trained for 5000 epochs on a T4 GPU in Google Colab.
Final Loss: 0.000042

Interactive 3D plot available here:
https://github.com/marwa2606/genesis-oracle/blob/main/data/pinn_3d_fabric.html

### Exercise 5: The Operator Horizon (FNOs)

- **Mapping Functional Spaces:** While a PINN learns a mapping 
from specific spatial-temporal coordinates (x,t) to a scalar u 
(requiring complete retraining for any new initial conditions), 
an FNO learns an operator mapping an entire input functional space 
directly to the continuous solution functional space u(x,t).

- **Frequency Domain Convolutions:** FNOs compute global integral 
operators using Fast Fourier Transforms (FFT). By filtering only 
the lower-frequency modes, the network captures global spatial 
dependencies far more efficiently than standard grid convolutions.

- **Zero-Shot Generalization:** Because FNO weights parameterize 
the continuous frequency domain, the model learns the underlying 
abstract mathematical operator and can perform instant zero-shot 
predictions for entirely new initial conditions without retraining.

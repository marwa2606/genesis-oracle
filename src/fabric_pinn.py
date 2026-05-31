import jax
import jax.numpy as jnp
import flax.linen as nn

class HeatSurrogate(nn.Module):
    """
    Physics-Informed Neural Network (PINN) surrogate model for the 1D Heat Equation.
    """
    @nn.compact
    def __call__(self, x):
        # x is expected to be a 2D continuous input containing (x, t) coordinates.
        
        # Why tanh instead of ReLU for PINNs?
        # -----------------------------------
        # PINNs embed the governing Partial Differential Equations (PDEs) directly into 
        # the loss function. This requires computing the PDE residuals, which involves 
        # taking exact derivatives of the network's output with respect to its inputs 
        # using automatic differentiation (e.g., d^2u/dx^2 for the heat equation).
        # 
        # ReLU is piecewise linear; its first derivative is a step function, and its 
        # second derivative is zero everywhere (and undefined at the origin). If we used 
        # ReLU, the second-order terms in the PDE residual would vanish completely, making 
        # it impossible to train the network to satisfy the physics.
        #
        # Therefore, we use a smooth, infinitely differentiable activation function like 
        # tanh (or Swish/GELU/Sine) which provides non-trivial, continuous higher-order 
        # derivatives.
        
        # 4 hidden layers with 32 neurons each
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        # Output layer: 1D scalar (temperature u)
        x = nn.Dense(1)(x)
        return x

def main():
    # ---------------------------------------------------------
    # Execution block
    # ---------------------------------------------------------
    
    # 1. Create a dummy input of shape (1, 2) representing one (x, t) point
    dummy_input = jnp.ones((1, 2))
    
    # 2. Instantiate the model
    model = HeatSurrogate()
    
    # 3. Initialize model with PRNGKey(0) and the dummy input
    rng_key = jax.random.PRNGKey(0)
    params = model.init(rng_key, dummy_input)
    
    # 4. Run a forward pass
    output = model.apply(params, dummy_input)
    
    # 5. Print output shape to verify
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()

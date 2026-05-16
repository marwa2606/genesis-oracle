"""
JAX & Flax Paradigm: Pure Functions and Explicit State
------------------------------------------------------
Why JAX forbids hidden state:
JAX is designed around pure functional programming principles to enable powerful 
transformations like jax.jit (compilation) and jax.grad (automatic differentiation). 
If a function has hidden, internal state (like updating an internal variable implicitly), 
it creates side effects. JAX transformations require deterministic, pure functions where the 
same inputs always yield the exact same outputs, making hidden state strictly forbidden.

How Flax solves this with explicit parameter trees:
Flax adheres to JAX's pure functional constraints by decoupling the model architecture 
from the model state (weights and biases). Instead of storing parameters inside the 
layer objects, Flax models act as pure functions. Parameters are initialized and 
returned as a completely separate nested dictionary structure (called a PyTree). 
These parameters must then be explicitly passed into the model for every forward pass.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

# 1. Define an MLP module with flax.linen
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model."""
    
    @nn.compact
    def __call__(self, x):
        # First hidden layer: 64 neurons with ReLU activation
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        
        # Second hidden layer: 64 neurons with ReLU activation
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        
        # Output layer: 1 neuron
        x = nn.Dense(features=1)(x)
        return x

def main():
    # 2. Heavily commented execution block
    
    # Define a pseudo-random number generator (PRNG) key for reproducible stochasticity
    key = jax.random.PRNGKey(0)
    
    # JAX requires splitting keys explicitly whenever random numbers are generated
    key_input, key_init = jax.random.split(key)
    
    # Create a random input tensor of shape (4, 8)
    # This represents a batch of 4 samples, each containing 8 features.
    x = jax.random.normal(key_input, (4, 8))
    
    # Instantiate the model class. 
    # Note: `model` currently contains NO weights or state! It only defines the architecture logic.
    model = MLP()
    
    # Initialize the model parameters. 
    # We pass in a PRNG key and a dummy input 'x' so Flax can automatically infer the required shapes.
    # The returned 'params' are stored completely OUTSIDE the model instance.
    #
    # === JAX/FLAX vs KERAS COMPARISON ===
    # Keras (Stateful): The model object internally stores its weights (e.g., accessed via model.weights).
    #                   You perform a forward pass by simply calling `output = model(x)`.
    # Flax (Stateless): The model object has no memory. Parameters live in a separate variable (`params`).
    #                   You perform a forward pass by explicitly injecting the parameters: `model.apply(params, x)`.
    # ====================================
    params = model.init(key_init, x)
    
    # Run the forward pass using model.apply
    # We explicitly provide both the architecture's parameters and the input data.
    output = model.apply(params, x)
    
    # Print the output shape and sample output
    print("--- Flax MLP Execution Results ---")
    print(f"Output Shape: {output.shape}")
    print(f"Sample Output Data:\n{output}")

if __name__ == "__main__":
    main()

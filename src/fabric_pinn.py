import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import plotly.graph_objects as go
import os

class HeatSurrogate(nn.Module):
    """
    Physics-Informed Neural Network (PINN) surrogate model for the 1D Heat Equation.
    """
    @nn.compact
    def __call__(self, x):
        # 4 hidden layers with 32 neurons each, using tanh
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

# Instantiate the global model
model = HeatSurrogate()

# Define alpha (thermal diffusivity)
alpha = 0.05

# Pure function to predict u for a single (x, t) point
def predict_u(params, x, t):
    inputs = jnp.array([x, t])
    u = model.apply(params, inputs)
    return u[0]

# Physics loss (PDE residual)
def physics_loss(params, x_pde, t_pde):
    def pde_residual(x, t):
        u_t = jax.grad(predict_u, argnums=2)(params, x, t)
        
        u_x_fn = jax.grad(predict_u, argnums=1)
        u_xx = jax.grad(u_x_fn, argnums=1)(params, x, t)
        
        return u_t - alpha * u_xx

    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    residuals = vmap_pde_residual(x_pde, t_pde)
    return jnp.mean(residuals**2)

# Initial Condition (IC) loss
def ic_loss(params, x_ic, t_ic, u_ic):
    vmap_predict = jax.vmap(predict_u, in_axes=(None, 0, 0))
    preds = vmap_predict(params, x_ic, t_ic)
    preds = preds.reshape(u_ic.shape)
    return jnp.mean((preds - u_ic)**2)

# Boundary Condition (BC) loss
def bc_loss(params, x_bc, t_bc, u_bc):
    vmap_predict = jax.vmap(predict_u, in_axes=(None, 0, 0))
    preds = vmap_predict(params, x_bc, t_bc)
    preds = preds.reshape(u_bc.shape)
    return jnp.mean((preds - u_bc)**2)

# Total loss function
def total_loss(params, data):
    pde_loss = physics_loss(params, data['pde']['x'], data['pde']['t'])
    i_loss = ic_loss(params, data['ic']['x'], data['ic']['t'], data['ic']['u'])
    b_loss = bc_loss(params, data['bc']['x'], data['bc']['t'], data['bc']['u'])
    return pde_loss + i_loss + b_loss

# Value and grad function
loss_grad_fn = jax.value_and_grad(total_loss)

# Optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Training step wrapped in jax.jit for speed
@jax.jit
def train_step(params, opt_state, data):
    loss, grads = loss_grad_fn(params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def generate_data():
    JAX_KEY = jax.random.PRNGKey(42)
    key_pde, key_ic, key_bc_left, key_bc_right = jax.random.split(JAX_KEY, 4)
    
    key_pde_x, key_pde_t = jax.random.split(key_pde, 2)
    x_pde = jax.random.uniform(key_pde_x, (5000,), minval=-1.0, maxval=1.0)
    t_pde = jax.random.uniform(key_pde_t, (5000,), minval=0.0, maxval=1.0)
    
    x_ic = jax.random.uniform(key_ic, (500,), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((500,))
    u_ic = -jnp.sin(jnp.pi * x_ic)
    
    t_bc_left = jax.random.uniform(key_bc_left, (250,), minval=0.0, maxval=1.0)
    x_bc_left = -jnp.ones((250,))
    u_bc_left = jnp.zeros((250,))
    
    t_bc_right = jax.random.uniform(key_bc_right, (250,), minval=0.0, maxval=1.0)
    x_bc_right = jnp.ones((250,))
    u_bc_right = jnp.zeros((250,))
    
    x_bc = jnp.concatenate([x_bc_left, x_bc_right])
    t_bc = jnp.concatenate([t_bc_left, t_bc_right])
    u_bc = jnp.concatenate([u_bc_left, u_bc_right])
    
    return {
        'pde': {'x': x_pde, 't': t_pde},
        'ic': {'x': x_ic, 't': t_ic, 'u': u_ic},
        'bc': {'x': x_bc, 't': t_bc, 'u': u_bc}
    }

def main():
    # Load dataset
    data = generate_data()
    
    # Initialize model
    key_init = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((2,))
    params = model.init(key_init, dummy_input)
    opt_state = optimizer.init(params)
    
    # Training Loop
    epochs = 5000
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        params, opt_state, loss = train_step(params, opt_state, data)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
    print(f"Final Loss at Epoch {epochs}: {loss:.6f}")
    
    # Generate predictions for visualization
    print("Generating predictions for 3D visualization...")
    
    # Create meshgrid: 100 points for x in [-1,1] and 100 points for t in [0,1]
    x_grid = jnp.linspace(-1, 1, 100)
    t_grid = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x_grid, t_grid)
    
    # Flatten to (10000,) inputs
    X_flat = X.flatten()
    T_flat = T.flatten()
    
    # Run through trained model to get U predictions
    vmap_predict = jax.vmap(predict_u, in_axes=(None, 0, 0))
    U_flat = vmap_predict(params, X_flat, T_flat)
    
    # Reshape U back to (100, 100)
    U = U_flat.reshape((100, 100))
    
    # Save interactive 3D plot
    os.makedirs('data', exist_ok=True)
    out_path = 'data/pinn_3d_fabric.html'
    print(f"Saving interactive 3D plot to {out_path}...")
    
    fig = go.Figure(data=[go.Surface(
        z=U, 
        x=X, 
        y=T, 
        colorscale='inferno'
    )])
    
    fig.update_layout(
        title='PINN: 1D Heat Equation Solution',
        scene=dict(
            xaxis_title='Space (x)',
            yaxis_title='Time (t)',
            zaxis_title='Temperature u(x,t)'
        )
    )
    
    fig.write_html(out_path)
    print("Done!")

if __name__ == "__main__":
    main()

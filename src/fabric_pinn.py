import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

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

# 1. Define alpha (thermal diffusivity)
alpha = 0.05

# 2. Pure function to predict u for a single (x, t) point
def predict_u(params, x, t):
    """
    Takes params, scalar x, and scalar t.
    Returns scalar temperature u.
    """
    # Combine scalars into a single 1D array of shape (2,)
    inputs = jnp.array([x, t])
    u = model.apply(params, inputs)
    # u is of shape (1,), extract scalar
    return u[0]

# 3. Physics loss (PDE residual)
def physics_loss(params, x_pde, t_pde):
    """
    Computes PDE residual: (u_t - alpha * u_xx)^2
    Vectorizes over all points using jax.vmap.
    """
    # Define the residual for a single point
    def pde_residual(x, t):
        # First derivative w.r.t t (argnums=2 because args are: params, x, t)
        u_t = jax.grad(predict_u, argnums=2)(params, x, t)
        
        # First derivative w.r.t x (argnums=1)
        u_x_fn = jax.grad(predict_u, argnums=1)
        # Second derivative w.r.t x
        u_xx = jax.grad(u_x_fn, argnums=1)(params, x, t)
        
        return u_t - alpha * u_xx

    # Vectorize over the spatial and temporal points (arrays of scalars)
    vmap_pde_residual = jax.vmap(pde_residual, in_axes=(0, 0))
    
    residuals = vmap_pde_residual(x_pde, t_pde)
    return jnp.mean(residuals**2)

# 4. Initial Condition (IC) loss
def ic_loss(params, x_ic, t_ic, u_ic):
    """
    Computes MSE between model prediction and u_true at t=0.
    """
    vmap_predict = jax.vmap(predict_u, in_axes=(None, 0, 0))
    preds = vmap_predict(params, x_ic, t_ic)
    # Ensure preds has same shape as u_ic
    preds = preds.reshape(u_ic.shape)
    return jnp.mean((preds - u_ic)**2)

# 5. Boundary Condition (BC) loss
def bc_loss(params, x_bc, t_bc, u_bc):
    """
    Computes MSE between model prediction and u_true at boundaries.
    """
    vmap_predict = jax.vmap(predict_u, in_axes=(None, 0, 0))
    preds = vmap_predict(params, x_bc, t_bc)
    preds = preds.reshape(u_bc.shape)
    return jnp.mean((preds - u_bc)**2)

# 6. Total loss function
def total_loss(params, data):
    """
    Returns the sum of physics, IC, and BC losses.
    """
    pde_loss = physics_loss(params, data['pde']['x'], data['pde']['t'])
    i_loss = ic_loss(params, data['ic']['x'], data['ic']['t'], data['ic']['u'])
    b_loss = bc_loss(params, data['bc']['x'], data['bc']['t'], data['bc']['u'])
    return pde_loss + i_loss + b_loss

def main():
    # Setup some dummy dataset (mimicking pinn_data.py but with arbitrary numbers for testing)
    key = jax.random.PRNGKey(42)
    key, key_init = jax.random.split(key)
    
    # Dummy params
    dummy_input = jnp.ones((2,))
    params = model.init(key_init, dummy_input)
    
    # Collocation points (PDE)
    x_pde = jax.random.uniform(key, (5000,), minval=-1.0, maxval=1.0)
    t_pde = jax.random.uniform(key, (5000,), minval=0.0, maxval=1.0)
    
    # Initial Condition
    x_ic = jax.random.uniform(key, (500,), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((500,))
    u_ic = -jnp.sin(jnp.pi * x_ic)
    
    # Boundary Conditions
    t_bc_left = jax.random.uniform(key, (250,), minval=0.0, maxval=1.0)
    x_bc_left = -jnp.ones((250,))
    u_bc_left = jnp.zeros((250,))
    
    t_bc_right = jax.random.uniform(key, (250,), minval=0.0, maxval=1.0)
    x_bc_right = jnp.ones((250,))
    u_bc_right = jnp.zeros((250,))
    
    x_bc = jnp.concatenate([x_bc_left, x_bc_right])
    t_bc = jnp.concatenate([t_bc_left, t_bc_right])
    u_bc = jnp.concatenate([u_bc_left, u_bc_right])
    
    data = {
        'pde': {'x': x_pde, 't': t_pde},
        'ic': {'x': x_ic, 't': t_ic, 'u': u_ic},
        'bc': {'x': x_bc, 't': t_bc, 'u': u_bc}
    }
    
    print("Computing initial losses...")
    pde_l = physics_loss(params, x_pde, t_pde)
    i_l = ic_loss(params, x_ic, t_ic, u_ic)
    b_l = bc_loss(params, x_bc, t_bc, u_bc)
    t_l = total_loss(params, data)
    
    print(f"Physics Loss (PDE) : {pde_l:.6f}")
    print(f"Initial Cond Loss  : {i_l:.6f}")
    print(f"Boundary Cond Loss : {b_l:.6f}")
    print(f"Total Loss         : {t_l:.6f}")
    
    # 7. Add basic Optax Adam optimizer setup with learning_rate=1e-3
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    print("\nOptimizer initialized successfully.")

if __name__ == "__main__":
    main()

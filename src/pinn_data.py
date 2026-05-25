import jax
import jax.numpy as jnp

def main():
    # Master key
    JAX_KEY = jax.random.PRNGKey(42)
    
    # Split the master key into subkeys for each generation task
    key_pde, key_ic, key_bc_left, key_bc_right = jax.random.split(JAX_KEY, 4)
    
    # -------------------------------------------------------------------
    # 1. Collocation Points (PDE)
    # 5000 random (x, t) points anywhere inside the domain
    # Domain: x in [-1, 1], t in [0, 1]
    # -------------------------------------------------------------------
    key_pde_x, key_pde_t = jax.random.split(key_pde, 2)
    x_pde = jax.random.uniform(key_pde_x, (5000, 1), minval=-1.0, maxval=1.0)
    t_pde = jax.random.uniform(key_pde_t, (5000, 1), minval=0.0, maxval=1.0)
    pde_points = jnp.hstack([x_pde, t_pde])
    
    # -------------------------------------------------------------------
    # 2. Initial Condition (IC)
    # 500 points at t=0
    # x sampled randomly in [-1, 1]
    # u_true = -jnp.sin(jnp.pi * x)
    # -------------------------------------------------------------------
    x_ic = jax.random.uniform(key_ic, (500, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((500, 1))
    ic_points = jnp.hstack([x_ic, t_ic])
    ic_u_true = -jnp.sin(jnp.pi * x_ic)
    
    # -------------------------------------------------------------------
    # 3. Boundary Conditions (BC)
    # 500 points total
    # 250 points at x=-1 with random t in [0,1], u_true = 0
    # 250 points at x=+1 with random t in [0,1], u_true = 0
    # -------------------------------------------------------------------
    t_bc_left = jax.random.uniform(key_bc_left, (250, 1), minval=0.0, maxval=1.0)
    x_bc_left = -jnp.ones((250, 1))
    bc_left_points = jnp.hstack([x_bc_left, t_bc_left])
    bc_left_u_true = jnp.zeros((250, 1))
    
    t_bc_right = jax.random.uniform(key_bc_right, (250, 1), minval=0.0, maxval=1.0)
    x_bc_right = jnp.ones((250, 1))
    bc_right_points = jnp.hstack([x_bc_right, t_bc_right])
    bc_right_u_true = jnp.zeros((250, 1))
    
    # Combine the two boundaries
    bc_points = jnp.vstack([bc_left_points, bc_right_points])
    bc_u_true = jnp.vstack([bc_left_u_true, bc_right_u_true])
    
    # -------------------------------------------------------------------
    # Print the shape of each dataset to verify
    # -------------------------------------------------------------------
    print("Collocation Points (PDE):")
    print(f"  Points (x, t) shape: {pde_points.shape}")
    
    print("\nInitial Condition (IC):")
    print(f"  Points (x, t) shape: {ic_points.shape}")
    print(f"  u_true shape: {ic_u_true.shape}")
    
    print("\nBoundary Conditions (BC):")
    print(f"  Points (x, t) shape: {bc_points.shape}")
    print(f"  u_true shape: {bc_u_true.shape}")

if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp

def projectile_loss(v_initial):
    """
    Simulates the horizontal distance of a projectile and returns the Mean Squared Error (MSE) 
    compared to a target distance of 150.0.
    
    The simulation uses a 45 degree angle (pi/4 radians).
    Distance formula: d = (v_initial^2 * sin(2 * theta)) / g
    """
    # 1. Calculate horizontal distance
    # Note: sin(2 * pi/4) = sin(pi/2) = 1.0
    distance = (v_initial**2 * jnp.sin(2 * jnp.pi / 4)) / 9.81
    
    # Return MSE against the target distance
    return (distance - 150.0)**2

def main():
    # 2. Use jax.grad to create a gradient function
    # jax.grad automatically computes the exact analytical derivative 
    # of projectile_loss with respect to its input parameter (v_initial).
    loss_gradient_fn = jax.grad(projectile_loss)
    
    # 3. Gradient descent loop
    v_initial = 10.0
    learning_rate = 0.1
    iterations = 20
    
    print("Starting gradient descent optimization...")
    
    for i in range(iterations):
        # Compute the current loss and gradient
        loss = projectile_loss(v_initial)
        gradient = loss_gradient_fn(v_initial)
        
        # Print v and loss at each iteration
        print(f"Iteration {i+1:2d} | v_initial = {v_initial:12.4f} | Loss = {loss:15.4f} | Gradient = {gradient:15.4f}")
        
        # Update rule
        v_initial = v_initial - learning_rate * gradient

    # 4. Print the final optimized v_initial at the end
    print("\nOptimization Complete!")
    print(f"Final optimized v_initial: {v_initial:.4f}")

if __name__ == "__main__":
    main()

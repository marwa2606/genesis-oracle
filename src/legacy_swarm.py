import numpy as np
import time

def simulate_swarm():
    """
    Simulates 100,000 damped harmonic oscillators over 1,000 time steps 
    using explicit Euler integration.
    """
    num_oscillators = 100000
    num_steps = 1000
    dt = 0.01
    damping_coef = 0.1
    
    # Initialize arrays
    positions = np.zeros(num_oscillators)
    velocities = np.ones(num_oscillators)  # Initialize with non-zero velocities
    omegas = np.random.uniform(0.5, 2.0, num_oscillators)  # Random frequencies
    
    # Explicit Euler integration
    for _ in range(num_steps):
        # Calculate accelerations (a = -omega^2 * x - c * v)
        accelerations = -(omegas ** 2) * positions - damping_coef * velocities
        
        # Update positions and velocities
        positions += velocities * dt
        velocities += accelerations * dt
        
    return positions, velocities

if __name__ == "__main__":
    print("Starting simulation...")
    start_time = time.time()
    
    simulate_swarm()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Total execution time: {execution_time:.4f} seconds")

import numpy as np
import matplotlib.pyplot as plt
import time

def estimate_pi(n_points=5_000_000):
    print(f"Estimating pi with {n_points:,} points...")
    start_time = time.perf_counter()
    
    # Generate random (x, y) points
    points = np.random.uniform(0, 1, size=(n_points, 2))
    
    # Check which points are inside the quarter circle
    # x**2 + y**2 <= 1
    dist_sq = np.sum(points**2, axis=1)
    inside = dist_sq <= 1
    n_inside = np.sum(inside)
    
    pi_estimate = 4 * n_inside / n_points
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    print(f"Estimated pi: {pi_estimate}")
    print(f"Execution time: {execution_time:.4f} seconds")
    
    return pi_estimate, execution_time

def create_plot(n_plot_points=10_000):
    print(f"Creating scatter plot with {n_plot_points:,} points...")
    points = np.random.uniform(0, 1, size=(n_plot_points, 2))
    dist_sq = np.sum(points**2, axis=1)
    inside = dist_sq <= 1
    
    plt.figure(figsize=(8, 8))
    
    # Scatter plot for points
    plt.scatter(points[inside, 0], points[inside, 1], color='blue', s=1, label='Inside')
    plt.scatter(points[~inside, 0], points[~inside, 1], color='red', s=1, label='Outside')
    
    # Draw the quarter circle boundary
    theta = np.linspace(0, np.pi/2, 100)
    plt.plot(np.cos(theta), np.sin(theta), color='black', linewidth=2)
    
    plt.title("Monte Carlo Estimation of Pi")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = "data/classical_pi_disp.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    estimate_pi()
    create_plot()

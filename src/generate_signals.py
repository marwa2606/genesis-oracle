import os
import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Generate a clean sine wave signal (500 time steps)
    n_steps = 500
    x = np.arange(n_steps)
    # Generate a nice sine wave (e.g. 5 periods)
    y = np.sin(2 * np.pi * 5 * x / n_steps)
    
    # 2. Inject a clipping artifact: set amplitude to max value (saturation)
    # at a random timestep between 150-350 for exactly 20 timesteps.
    # The max value of the sine wave is 1.0.
    start_idx = random.randint(150, 350)
    y[start_idx : start_idx + 20] = 1.0
    
    # 3. Create a beautiful premium-looking plot with matplotlib
    plt.figure(figsize=(10, 5), dpi=300)
    plt.style.use('dark_background')  # Sleek dark mode
    
    # Color palette: HSL tailored vibrant gradient/color
    # A beautiful neon cyan for the signal line
    plt.plot(x, y, color='#00F5FF', linewidth=2, label='Signal Stream')
    
    # Customize grid and labels
    plt.grid(True, linestyle='--', color='#223344', alpha=0.7)
    plt.title("Sensor Stream Telemetry System Audit", fontsize=14, fontweight='bold', pad=15, color='#F0F8FF')
    plt.xlabel("Timeline (Timesteps)", fontsize=11, color='#AFEEEE')
    plt.ylabel("Normalized Amplitude", fontsize=11, color='#AFEEEE')
    plt.xlim(0, n_steps)
    plt.ylim(-1.2, 1.2)
    plt.legend(loc='upper right', framealpha=0.5, facecolor='#112233', edgecolor='#00F5FF')
    
    # Tight layout to avoid clipping margins
    plt.tight_layout()
    
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save as data/audit_target.png
    plt.savefig("data/audit_target.png", facecolor='#0D1117', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    main()

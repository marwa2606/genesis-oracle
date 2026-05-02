import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # 1. Generate Fourier square wave signal parameters
    T = 22.0
    omega_0 = 2 * np.pi / T
    num_periods = 100
    points_per_period = 500
    total_points = num_periods * points_per_period
    
    # Time array for 100 periods
    t = np.linspace(0, num_periods * T, total_points, endpoint=False)
    
    # 2. Apply RC low-pass filter analytically
    R = 500.0
    C = (1000 + 322) * 1e-6  # Farads
    
    # Generate the baseline signal from the first 9 odd harmonics
    baseline_signal = np.zeros_like(t)
    odd_harmonics = [2*i + 1 for i in range(9)] # 1, 3, 5, 7, 9, 11, 13, 15, 17
    
    for n in odd_harmonics:
        omega_n = n * omega_0
        
        # H(omega) = 1 / (1 + j*omega*R*C)
        H = 1.0 / (1.0 + 1j * omega_n * R * C)
        
        # The original amplitude for square wave is 4/(pi*n)
        # Apply the filter's magnitude and phase shift to the harmonic
        A_n = (4.0 / (np.pi * n)) * np.abs(H)
        phi_n = np.angle(H)
        
        baseline_signal += A_n * np.sin(omega_n * t + phi_n)
        
    # 3. Add random Gaussian noise
    np.random.seed(42)  # For reproducible noise
    noise_amplitude = 0.05
    noise = np.random.normal(0, noise_amplitude, size=total_points)
    signal = baseline_signal + noise
    
    # Inject massive high-frequency voltage spike between period 70 and 75
    spike_mask = (t >= 70 * T) & (t <= 75 * T)
    
    # High-frequency burst (10 Hz is much higher than base freq 1/22 Hz)
    f_spike = 10.0  
    spike_burst = 5.0 * np.sin(2 * np.pi * f_spike * t)
    
    # Smooth envelope for the spike burst to blend it over the 5 periods
    envelope = np.sin(np.pi * (t - 70 * T) / (5 * T)) ** 2
    
    # Add the anomaly to the signal
    signal[spike_mask] += (spike_burst * envelope)[spike_mask]
    
    # 4. Save the full signal
    os.makedirs('data', exist_ok=True)
    np.save('data/signal.npy', signal)
    print("Saved signal to data/signal.npy")
    
    # 5. Plot two windows side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: normal noisy signal (period 1-10) -> t in [0, 10*T]
    window1_mask = (t >= 0) & (t <= 10 * T)
    axes[0].plot(t[window1_mask], signal[window1_mask], color='#1f77b4', linewidth=1.5)
    axes[0].set_title('Normal Noisy Signal (Periods 1-10)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (V)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Right: anomaly spike (period 68-77) -> t in [68*T, 77*T]
    window2_mask = (t >= 68 * T) & (t <= 77 * T)
    axes[1].plot(t[window2_mask], signal[window2_mask], color='#d62728', linewidth=1.5)
    axes[1].set_title('Anomaly Spike (Periods 68-77)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (V)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('data/data_feed.png', dpi=300)
    print("Saved plot to data/data_feed.png")

if __name__ == "__main__":
    main()

import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import matplotlib.pyplot as plt
import keras
from architecture_conv import PhysicsAutoencoder, create_windows

def main():
    # Load data
    signal = np.load('data/signal.npy')
    
    # Split the data
    points_per_period = 500
    split_index = 60 * points_per_period
    
    train_signal = signal[:split_index]
    test_signal = signal[split_index:]
    
    # Create windows
    window_size = 50
    X_train = create_windows(train_signal, window_size=window_size)
    X_test = create_windows(test_signal, window_size=window_size)
    
    # Initialize and compile model
    model = PhysicsAutoencoder()
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    print("Training Conv1D Autoencoder...")
    model.fit(X_train, X_train, epochs=5, batch_size=128, validation_split=0.1)
    
    # Predict
    print("Running predictions on test set...")
    X_pred = model.predict(X_test)
    
    # Calculate MSE
    mse = np.mean(np.square(X_test - X_pred), axis=1)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(mse, color='crimson', label='Reconstruction Error (MSE)')
    plt.title('Conv1D Autoencoder Anomaly Detection')
    plt.xlabel('Window Index (Test Set)')
    plt.ylabel('Mean Squared Error')
    
    # Highlight anomaly region: test_signal starts at period 60.
    # Anomaly was injected at period 70 to 75. So it's 10 to 15 periods into the test set.
    # Each period is 500 points.
    plt.axvspan(10*500, 15*500, color='orange', alpha=0.3, label='Known Spike Region')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save
    plt.savefig('anomaly_detection.png', dpi=300)
    print("Saved anomaly_detection.png to root directory.")

if __name__ == "__main__":
    main()

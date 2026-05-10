import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import keras
from keras import ops

def create_windows(signal, window_size=50):
    """
    Slices a 1D numpy array into overlapping 2D windows.
    """
    windows = []
    for i in range(len(signal) - window_size + 1):
        windows.append(signal[i:i + window_size])
    return np.array(windows)

class SignalCompression(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SignalCompression, self).__init__(**kwargs)
        self.reshape = keras.layers.Reshape((-1, 1))
        # Compress spatial dimension by 5
        self.conv1 = keras.layers.Conv1D(
            filters=16, kernel_size=5, strides=5, activation="relu", padding="same", name="conv1d_1"
        )
        # Compress spatial dimension by 2
        self.conv2 = keras.layers.Conv1D(
            filters=8, kernel_size=2, strides=2, activation="relu", padding="same", name="conv1d_2"
        )

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SignalExpansion(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SignalExpansion, self).__init__(**kwargs)
        # Expand spatial dimension by 2
        self.convT1 = keras.layers.Conv1DTranspose(
            filters=16, kernel_size=2, strides=2, activation="relu", padding="same", name="conv1d_transpose_1"
        )
        # Expand spatial dimension by 5
        self.convT2 = keras.layers.Conv1DTranspose(
            filters=1, kernel_size=5, strides=5, activation="linear", padding="same", name="conv1d_transpose_2"
        )
        self.flatten = keras.layers.Flatten()

    def call(self, inputs):
        x = self.convT1(inputs)
        x = self.convT2(x)
        return self.flatten(x)

class PhysicsAutoencoder(keras.Model):
    def __init__(self, **kwargs):
        super(PhysicsAutoencoder, self).__init__(**kwargs)
        self.encoder = SignalCompression()
        self.decoder = SignalExpansion()

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    # Load data
    signal = np.load('data/signal.npy')
    
    # Split the data
    # Assuming period is 500 based on data_generator.py (total_points = num_periods * points_per_period)
    # normal data before period 60 for training, rest for testing.
    points_per_period = 500
    split_index = 60 * points_per_period
    
    train_signal = signal[:split_index]
    test_signal = signal[split_index:]
    
    # Create windows
    window_size = 50
    X_train = create_windows(train_signal, window_size=window_size)
    X_test = create_windows(test_signal, window_size=window_size)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Model summary
    model = PhysicsAutoencoder()
    model.build((None, window_size))
    model.summary()

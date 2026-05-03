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
    def __init__(self, output_dim=8, **kwargs):
        super(SignalCompression, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="random_normal",
            trainable=True,
            name='compression_matrix'
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name='compression_bias'
        )

    def call(self, inputs):
        return ops.relu(ops.matmul(inputs, self.w) + self.b)

class SignalExpansion(keras.layers.Layer):
    def __init__(self, output_dim=50, **kwargs):
        super(SignalExpansion, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="random_normal",
            trainable=True,
            name='expansion_matrix'
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name='expansion_bias'
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

class PhysicsAutoencoder(keras.Model):
    def __init__(self, **kwargs):
        super(PhysicsAutoencoder, self).__init__(**kwargs)
        self.encoder = SignalCompression(output_dim=8)
        self.decoder = SignalExpansion(output_dim=50)

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

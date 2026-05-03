import os
import numpy as np
import keras
from keras import layers, Model
from keras import ops

def create_windows(signal, window_size=50):
    """
    Slices a 1D numpy array into overlapping 2D windows.
    """
    if len(signal) < window_size:
        return np.empty((0, window_size))
        
    num_windows = len(signal) - window_size + 1
    # Using stride tricks for memory efficiency and speed
    shape = (num_windows, window_size)
    strides = (signal.strides[0], signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    # Copy to ensure the returned array is memory contiguous
    return windows.copy()

class SignalCompression(layers.Layer):
    """
    Custom Keras layer to compress windowed signals.
    """
    def __init__(self, output_dim=8, **kwargs):
        super(SignalCompression, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # Create a trainable weight matrix
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="compression_w"
        )
        # Create a trainable bias vector
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="compression_b"
        )

    def call(self, inputs):
        # Forward pass with ReLU activation using keras.ops
        return ops.relu(ops.matmul(inputs, self.w) + self.b)

class SignalExpansion(layers.Layer):
    """
    Custom Keras layer to expand compressed signals back to original size.
    """
    def __init__(self, output_dim=50, **kwargs):
        super(SignalExpansion, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="expansion_w"
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="expansion_b"
        )

    def call(self, inputs):
        # Linear activation for reconstruction using keras.ops
        return ops.matmul(inputs, self.w) + self.b

class PhysicsAutoencoder(Model):
    """
    Custom Keras model assembling the autoencoder structure.
    """
    def __init__(self, input_dim=50, latent_dim=8, **kwargs):
        super(PhysicsAutoencoder, self).__init__(**kwargs)
        self.encoder = SignalCompression(output_dim=latent_dim)
        self.decoder = SignalExpansion(output_dim=input_dim)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def main():
    signal_path = 'data/signal.npy'
    if not os.path.exists(signal_path):
        print(f"Error: {signal_path} not found. Please run the data generator first.")
        return
        
    signal = np.load(signal_path)
    
    # Split data: normal data before period 60 for training, rest for testing.
    # Based on T=22, 500 points per period: period 60 ends at index 60 * 500 = 30000.
    split_idx = 60 * 500
    
    train_signal = signal[:split_idx]
    test_signal = signal[split_idx:]
    
    # Create windows
    window_size = 50
    X_train = create_windows(train_signal, window_size=window_size)
    X_test = create_windows(test_signal, window_size=window_size)
    
    print("Data splitting and windowing complete.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}\n")
    
    # Initialize model
    model = PhysicsAutoencoder(input_dim=50, latent_dim=8)
    
    # Build model by passing a dummy input so the weights are created
    dummy_input = ops.zeros((1, 50))
    model(dummy_input)
    
    # Print the model summary
    print("Model Summary:")
    model.summary()

if __name__ == "__main__":
    main()

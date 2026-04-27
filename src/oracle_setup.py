import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax.numpy as jnp

# Zufälligen Tensor erstellen
tensor = keras.random.normal(shape=(3, 3))
print("Backend:", keras.backend.backend())
print("Tensor:", tensor)
print("Typ:", type(tensor))
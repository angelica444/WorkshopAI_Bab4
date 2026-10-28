"""Example 4.21: Create and summarize a simple Keras LSTM model."""

from tensorflow import keras
from tensorflow.keras import layers


if __name__ == "__main__":
    model = keras.Sequential(
        [
            layers.Embedding(input_dim=100, output_dim=32),
            layers.LSTM(64),
            layers.Dense(1),
        ]
    )
    model.summary()
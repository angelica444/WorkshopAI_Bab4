"""Example 4.18: Apply a Keras SimpleRNN layer to random sequences."""

import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    batch_size = 4
    time_steps = 5
    features = 1
    inputs = np.random.random((batch_size, time_steps, features)).astype(np.float32)
    simple_rnn = tf.keras.layers.SimpleRNN(4)
    print("Inputs:")
    print(inputs)
    print("RNN output:")
    print(simple_rnn(inputs))
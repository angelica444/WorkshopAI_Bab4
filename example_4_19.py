"""Example 4.19: Build and summarize a SimpleRNN classifier."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam


if __name__ == "__main__":
    model = Sequential(
        [
            SimpleRNN(50, input_shape=(49, 1)),
            Dense(46),
            Activation("softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
    model.summary()
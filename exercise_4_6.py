"""Exercise 4.6: Add a convolution, pooling, and dropout block to Example 4.7."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


def build_model(additional_block):
    layers = [Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)), MaxPooling2D(), Dropout(0.2)]
    if additional_block:
        layers.extend([Conv2D(32, (3, 3), activation="relu"), MaxPooling2D(), Dropout(0.2)])
    layers.extend([Flatten(), Dense(128, activation="relu"), Dense(2, activation="softmax")])
    return Sequential(layers)


if __name__ == "__main__":
    for added in (False, True):
        model = build_model(added)
        print(f"Additional block: {added}; parameters: {model.count_params():,}")
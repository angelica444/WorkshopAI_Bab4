"""Example 4.16: Train an autoencoder to reconstruct MNIST digits."""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape


def build_autoencoder(image_shape, code_size):
    encoder = Sequential([Input(shape=image_shape), Flatten(), Dense(code_size)], name="encoder")
    decoder = Sequential(
        [Input(shape=(code_size,)), Dense(int(np.prod(image_shape))), Reshape(image_shape)],
        name="decoder",
    )
    return encoder, decoder


def main():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    encoder, decoder = build_autoencoder(x_train[0].shape, 32)
    inputs = Input(shape=x_train[0].shape)
    autoencoder = Model(inputs, decoder(encoder(inputs)))
    autoencoder.compile(optimizer="adamax", loss="mse")
    autoencoder.summary()
    history = autoencoder.fit(x_train, x_train, epochs=20, validation_data=(x_test, x_test))

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Autoencoder loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    reconstructions = autoencoder.predict(x_test[:5], verbose=0)
    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for index in range(5):
        axes[0, index].imshow(x_test[index], cmap="gray")
        axes[1, index].imshow(reconstructions[index], cmap="gray")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
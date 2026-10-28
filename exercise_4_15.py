"""Exercise 4.15: Visualize filters from the second VGG19 convolution layer."""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19


if __name__ == "__main__":
    model = VGG19(weights="imagenet")
    layer_index = 2
    filters, biases = model.layers[layer_index].get_weights()
    print("Layer:", model.layers[layer_index].name)
    print("Filter shape:", filters.shape, "bias shape:", biases.shape)
    value_min, value_max = filters.min(), filters.max()
    filters = (filters - value_min) / (value_max - value_min + 1e-8)
    figure, axes = plt.subplots(4, 3, figsize=(6, 8))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        axis.imshow(filters[:, :, :, index], interpolation="nearest")
    plt.tight_layout()
    plt.show()
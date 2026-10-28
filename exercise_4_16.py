"""Exercise 4.16: Display every filter in VGG19's first convolution layer."""

import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19


if __name__ == "__main__":
    model = VGG19(weights="imagenet")
    filters, _ = model.layers[1].get_weights()
    value_min, value_max = filters.min(), filters.max()
    filters = (filters - value_min) / (value_max - value_min + 1e-8)
    filter_count = filters.shape[-1]
    figure, axes = plt.subplots(filter_count, 3, figsize=(6, filter_count * 1.5))
    for filter_index in range(filter_count):
        for channel in range(3):
            axes[filter_index, channel].imshow(filters[:, :, channel, filter_index], cmap="gray")
            axes[filter_index, channel].axis("off")
    plt.tight_layout()
    plt.show()
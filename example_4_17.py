"""Example 4.17: Inspect VGG19 layers, filters, and feature maps."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def plot_feature_maps(feature_maps):
    channel_count = feature_maps.shape[-1]
    columns = 8
    rows = (channel_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2 * rows), squeeze=False)
    for channel, axis in enumerate(axes.flat):
        axis.axis("off")
        if channel < channel_count:
            axis.imshow(feature_maps[0, :, :, channel], cmap="gray")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="Elephant.jpg")
    parser.add_argument("--layer", type=int, default=1, help="Layer index to visualize")
    args = parser.parse_args()

    model = VGG19(weights="imagenet")
    model.summary()
    for index, layer in enumerate(model.layers):
        print(index, layer.name)
        if "conv" in layer.name:
            filters, biases = layer.get_weights()
            print("  filter shape:", filters.shape, "bias shape:", biases.shape)

    filters, _ = model.layers[1].get_weights()
    value_min, value_max = filters.min(), filters.max()
    normalized_filters = (filters - value_min) / (value_max - value_min + 1e-8)
    figure, axes = plt.subplots(4, 3, figsize=(6, 8))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        axis.imshow(normalized_filters[:, :, :, index], interpolation="nearest")
    plt.tight_layout()
    plt.show()

    feature_model = Model(inputs=model.inputs, outputs=model.layers[args.layer].output)
    image = load_img(args.image, target_size=(224, 224))
    batch = np.expand_dims(img_to_array(image), axis=0)
    feature_maps = feature_model.predict(preprocess_input(batch), verbose=0)
    print("Feature maps:", feature_maps.shape)
    plot_feature_maps(feature_maps)


if __name__ == "__main__":
    main()
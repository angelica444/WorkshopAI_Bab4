"""Exercise 4.17: Visualize feature maps from a selected VGG19 layer."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="Elephant.jpg")
    parser.add_argument("--layer", type=int, default=2, help="Try 2, 4, or 7 for later convolution layers")
    args = parser.parse_args()

    base_model = VGG19(weights="imagenet")
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[args.layer].output)
    image = load_img(args.image, target_size=(224, 224))
    batch = np.expand_dims(img_to_array(image), axis=0)
    feature_maps = model.predict(preprocess_input(batch), verbose=0)
    count = feature_maps.shape[-1]
    columns = 8
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(16, 2 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            axis.imshow(feature_maps[0, :, :, index], cmap="gray")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
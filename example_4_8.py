"""Example 4.8: Loading and Visualizing MNIST Dataset using Keras and Matplotlib."""

import matplotlib.pyplot as plt
from keras.datasets import mnist

# Memuat dataset MNIST (gambar digit angka tulisan tangan)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Membuat jendela plot dengan 1 baris dan 4 kolom untuk menampilkan 4 gambar pertama
plt.figure(figsize=(10, 3))

plt.subplot(1, 4, 1)
plt.imshow(X_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")

plt.subplot(1, 4, 2)
plt.imshow(X_train[1], cmap="gray")
plt.title(f"Label: {y_train[1]}")

plt.subplot(1, 4, 3)
plt.imshow(X_train[2], cmap="gray")
plt.title(f"Label: {y_train[2]}")

plt.subplot(1, 4, 4)
plt.imshow(X_train[3], cmap="gray")
plt.title(f"Label: {y_train[3]}")

# Menampilkan plot gambar
plt.tight_layout()
plt.show()
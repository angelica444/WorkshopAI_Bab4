"""Example 4.6: Inception V3 Pre-trained Model using TensorFlow/Keras."""

from tensorflow.keras.applications import inception_v3

# Inisialisasi model Inception V3 dengan bobot pre-trained ImageNet
model = inception_v3.InceptionV3(weights="imagenet")

# Menampilkan ringkasan arsitektur model di terminal
print(model.summary())
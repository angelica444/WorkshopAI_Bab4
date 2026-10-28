"""Exercise 4.9: VGG19 Pre-trained Model using TensorFlow/Keras."""

from tensorflow.keras.applications.vgg19 import VGG19

# Memuat arsitektur model VGG19 dengan bobot pre-trained ImageNet
model = VGG19(weights="imagenet")

# Menampilkan ringkasan arsitektur model di terminal
print(model.summary())
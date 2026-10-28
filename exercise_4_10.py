"""Exercise 4.10: VGG19 Image Classification."""

import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image

# 1. Memuat arsitektur VGG19 dengan bobot pre-trained ImageNet
model = VGG19(weights="imagenet")

# 2. Menyiapkan dan memuat gambar
img_path = "Elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))

# 3. Konversi gambar ke array dan preprocessing
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 4. Melakukan prediksi
predictions = model.predict(x)
results = decode_predictions(predictions, top=5)[0]

# 5. Menampilkan hasil prediksi secara rapi
print("\n" + "=" * 45)
print("     HASIL PREDIKSI EXERCISE 4.10 (VGG19)")
print("=" * 45)
for i, (imagenet_id, label, probability) in enumerate(results):
    persen = float(probability) * 100
    print(f" {i + 1}. {label:<25} : {persen:.2f}%")
print("=" * 45)
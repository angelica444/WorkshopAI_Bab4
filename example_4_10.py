"""Example 4.10 & Exercise: VGG16 Image Classification and Prediction."""

import argparse
import numpy as np
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image

# Menyiapkan argumen terminal dengan default 'Elephant.png'
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=False,
    default="Elephant.png",
    help="Path to the image",
)
args = ap.parse_args()

try:
    # 1. Memuat arsitektur VGG16 dengan bobot pre-trained ImageNet
    print("[INFO] Memuat model VGG16...")
    model = VGG16(weights="imagenet")

    # 2. Memuat dan mengubah ukuran gambar ke (224, 224) piksel
    print(f"[INFO] Memuat gambar: {args.image}")
    loaded_image = image.load_img(args.image, target_size=(224, 224))

    # 3. Mengubah gambar menjadi array numpy
    image_array = image.img_to_array(loaded_image)

    # 4. Menambah dimensi batch (dari (224, 224, 3) menjadi (1, 224, 224, 3))
    image_array = np.expand_dims(image_array, axis=0)

    # 5. Melakukan preprocessing khusus VGG16
    processed_image = preprocess_input(image_array)

    # 6. Menjalankan prediksi model
    print("[INFO] Melakukan klasifikasi gambar...")
    predictions = model.predict(processed_image)

    # 7. Mendapatkan 3 prediksi teratas beserta tingkat keyakinannya
    decoded_preds = decode_predictions(predictions, top=3)[0]

    print("\n--- HASIL PREDIKSI KLASIFIKASI ---")
    for i, (imagenet_id, label, probability) in enumerate(decoded_preds):
        print(f"{i + 1}. {label}: {probability * 100:.2f}%")

except FileNotFoundError:
    print(
        f"\n[ERROR] File gambar '{args.image}' tidak ditemukan di folder aktif!"
    )
    print("Pastikan file gambar sudah diletakkan di dalam folder Bab4.")
except Exception as e:
    print(f"\n[ERROR] Terjadi kesalahan: {e}")
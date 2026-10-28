
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg16, resnet50, mobilenet, inception_v3

# Pilih model: VGG16, ResNet50, MobileNet, atau InceptionV3
MODEL_NAME = "MobileNet"

if MODEL_NAME == "VGG16":
    model = vgg16.VGG16(weights="imagenet")
    preprocess = vgg16.preprocess_input
    image_size = 224

elif MODEL_NAME == "ResNet50":
    model = resnet50.ResNet50(weights="imagenet")
    preprocess = resnet50.preprocess_input
    image_size = 224

elif MODEL_NAME == "MobileNet":
    model = mobilenet.MobileNet(weights="imagenet")
    preprocess = mobilenet.preprocess_input
    image_size = 224

elif MODEL_NAME == "InceptionV3":
    model = inception_v3.InceptionV3(weights="imagenet")
    preprocess = inception_v3.preprocess_input
    image_size = 299

else:
    raise ValueError("Nama model tidak tersedia.")

print(f"Model yang digunakan: {MODEL_NAME}")
model.summary()

img_path = "Elephant.jpg"
img = image.load_img(img_path, target_size=(image_size, image_size))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess(x)

predictions = model.predict(x, verbose=0)
results = decode_predictions(predictions, top=5)

print("\nHasil klasifikasi gambar:")
for result in results[0]:
    print(f"{result[1]}: {result[2] * 100:.2f}%")

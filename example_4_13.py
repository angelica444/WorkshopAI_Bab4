
import cv2
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

image_size = 224
model = vgg16.VGG16(weights="imagenet")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam tidak dapat dibuka.")
    raise SystemExit

try:
    while camera.isOpened():
        ok, cam_frame = camera.read()

        if not ok:
            print("Gagal membaca gambar dari webcam.")
            break

        frame = cv2.resize(cam_frame, (image_size, image_size))
        numpy_image = img_to_array(frame)
        image_batch = np.expand_dims(numpy_image, axis=0)
        processed_image = vgg16.preprocess_input(image_batch.copy())

        predictions = model.predict(processed_image, verbose=0)
        label = decode_predictions(predictions, top=1)

        object_name = label[0][0][1]
        confidence = label[0][0][2] * 100

        text = f"VGG16: {object_name}, {confidence:.1f}%"
        cv2.putText(
            cam_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
        )

        cv2.imshow("Webcam Classification", cam_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

"""Exercise 4.11: Classify webcam frames with VGG19."""

import cv2
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array


def main():
    model = vgg19.VGG19(weights="imagenet")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            batch = np.expand_dims(img_to_array(cv2.resize(frame, (224, 224))), axis=0)
            predictions = model.predict(vgg19.preprocess_input(batch), verbose=0)
            name, confidence = decode_predictions(predictions, top=1)[0][0][1:]
            cv2.putText(frame, f"VGG19: {name}, {confidence:.1%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("Webcam classification", frame)
            if cv2.waitKey(30) == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
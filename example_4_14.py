"""Example 4.14: Classify webcam frames with a selectable Keras application."""

import argparse

import cv2
import numpy as np
from tensorflow.keras.applications import inception_v3, mobilenet, resnet50, vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array


MODELS = {
    "vgg16": (vgg16.VGG16, vgg16.preprocess_input, 224),
    "resnet50": (resnet50.ResNet50, resnet50.preprocess_input, 224),
    "mobilenet": (mobilenet.MobileNet, mobilenet.preprocess_input, 224),
    "inceptionv3": (inception_v3.InceptionV3, inception_v3.preprocess_input, 299),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="resnet50")
    args = parser.parse_args()
    model_factory, preprocess, image_size = MODELS[args.model]
    model = model_factory(weights="imagenet")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            resized = cv2.resize(frame, (image_size, image_size))
            batch = np.expand_dims(img_to_array(resized), axis=0)
            predictions = model.predict(preprocess(batch), verbose=0)
            name, confidence = decode_predictions(predictions, top=1)[0][0][1:]
            cv2.putText(frame, f"{args.model}: {name}, {confidence:.1%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("Webcam classification", frame)
            if cv2.waitKey(30) == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
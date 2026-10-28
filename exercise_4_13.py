"""Exercise 4.13: Choose a model-zoo architecture for webcam comparison."""

import argparse

import cv2
import numpy as np
from classification_models.keras import Classifiers
from tensorflow.keras.applications.imagenet_utils import decode_predictions


MODEL_SIZES = {"vgg16": 224, "resnet50": 224, "mobilenetv2": 224, "densenet201": 224, "inceptionv3": 299}


def main():
    parser = argparse.ArgumentParser(description="Run once per --model value to compare architectures")
    parser.add_argument("--model", choices=MODEL_SIZES, default="vgg16")
    args = parser.parse_args()
    classifier, preprocess = Classifiers.get(args.model)
    image_size = MODEL_SIZES[args.model]
    model = classifier(input_shape=(image_size, image_size, 3), weights="imagenet", classes=1000)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            resized = cv2.resize(frame, (image_size, image_size))
            batch = preprocess(np.expand_dims(np.asarray(resized), axis=0))
            predictions = model.predict(batch, verbose=0)
            name, confidence = decode_predictions(predictions, top=1)[0][0][1:]
            cv2.putText(frame, f"{args.model}: {name}, {confidence:.1%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Classification", frame)
            if cv2.waitKey(30) == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
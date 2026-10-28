"""Exercise 4.7: Modified CNN with 64 Filters and (3,3) Kernel Size on MNIST."""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Memuat dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Membentuk ulang dan normalisasi data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32") / 255.0

# One-hot encoding label
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Membangun model dengan 64 filter dan kernel (3, 3)
model = Sequential()
model.add(
    Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation="relu")
)  # Modifikasi di sini
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# Kompilasi model
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Menampilkan ringkasan parameter
model.summary()

# Melatih model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=200,
    verbose=2,
)

# Evaluasi model
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Error: {100 - scores[1] * 100:.2f}%")
print(f"CNN Accuracy: {scores[1] * 100:.2f}%")
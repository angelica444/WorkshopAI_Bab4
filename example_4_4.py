"""Example 4.4: LeNet-5 CNN Architecture using Keras."""

from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from keras.models import Sequential

# Membangun arsitektur model LeNet-5
model = Sequential()
model.add(
    Conv2D(
        filters=6,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 1),
    )
)
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=84, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# Menampilkan ringkasan arsitektur model di terminal
model.summary()
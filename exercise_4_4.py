"""Exercise 4.4: Modified LeNet-5 CNN with Different Number of Filters."""

from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from keras.models import Sequential

# Memodifikasi arsitektur LeNet-5 dengan jumlah filter yang berbeda (12 dan 32)
model = Sequential()
model.add(
    Conv2D(
        filters=12,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 1),
    )
)
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=84, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# Menampilkan ringkasan arsitektur model di terminal
model.summary()
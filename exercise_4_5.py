"""Exercise 4.5: Modified AlexNet Model with Different Dense Layer Units."""

import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

# Membuat model AlexNet dengan modifikasi jumlah unit Dense Layer
model = Sequential()

# 1st Convolutional Layer
model.add(
    Conv2D(
        filters=96,
        input_shape=(224, 224, 3),
        kernel_size=(11, 11),
        activation="relu",
        strides=(4, 4),
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# 2nd Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(11, 11),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# 3rd Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)

# 4th Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)

# 5th Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# Modifikasi Fully Connected (Dense) Layers dengan jumlah unit yang lebih kecil (2048, 2048, 500)
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(500, activation="relu"))
model.add(Dropout(0.4))

# Output Layer (tetap 17 kelas)
model.add(Dense(17, activation="softmax"))

# Menampilkan ringkasan arsitektur model
model.summary()

# Kompilasi model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer="adam",
    metrics=["accuracy"],
)
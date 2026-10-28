"""Example 4.7: Customized Deep Learning Model with Keras."""

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

# Membuat model kustom
model = Sequential()
model.add(
    Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu")
)
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Kompilasi model
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Menampilkan ringkasan arsitektur model di terminal
model.summary()
"""Example 4.3: Multiple Layer Perceptron Classification (Breast Cancer)."""

from keras import optimizers
from keras.layers import Activation, Dense
from keras.models import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Memuat dataset Breast Cancer
whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

# Membagi data menjadi data latih (70%) dan data uji (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7
)
features = X_train.shape[1]

# Membangun Model MLP 3-Layer menggunakan Keras Sequential
model = Sequential()
model.add(Dense(10, activation="relu", input_shape=[X_train.shape[1]]))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

# Menampilkan ringkasan arsitektur model dan konfigurasinya
print(model.summary())

# Kompilasi model dengan optimizer adam dan fungsi loss mean squared error
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

# Pelatihan model (training)
model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1,
)

# Evaluasi model menggunakan data uji
results = model.evaluate(X_test, y_test)
print("Loss: ", results[0])
print("MSE / Accuracy Metric: ", results[1])
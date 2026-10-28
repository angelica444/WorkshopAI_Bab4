"""Exercise 4.1: Perceptron for Logical OR Gate."""

import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inisialisasi bobot dan bias dengan nilai nol
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Proses pelatihan (training) perceptron
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Hitung fungsi aktivasi linier (z = w * x + b)
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Fungsi aktivasi step (threshold function)
                y_predicted = 1 if linear_output >= 0 else 0

                # Perbarui bobot dan bias jika terjadi kesalahan prediksi
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([1 if i >= 0 else 0 for i in linear_output])


if __name__ == "__main__":
    # Dataset untuk gerbang logika OR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    # Melatih model Perceptron
    perceptron = Perceptron(learning_rate=0.1, epochs=10)
    perceptron.fit(X, y)

    # Menguji hasil prediksi model
    predictions = perceptron.predict(X)

    print("--- Hasil Uji Gerbang Logika OR ---")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Prediksi Output: {predictions[i]}")

    print(f"\nBobot akhir (Weights): {perceptron.weights}")
    print(f"Bias akhir: {perceptron.bias}")
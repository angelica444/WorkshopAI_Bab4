"""Exercise 4.2: MLP Classifier with 3 Inputs."""

from sklearn.neural_network import MLPClassifier

# Dataset logika OR dengan 3 input (8 kombinasi kemungkinan)
X = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
]

# Target y untuk gerbang OR 3-input (hanya [0,0,0] yang bernilai 0, sisanya 1)
y = [0, 1, 1, 1, 1, 1, 1, 1]

# Inisialisasi model MLP dengan hidden layers (5, 2)
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)

# Melatih model menggunakan data baru
clf.fit(X, y)

# Menguji prediksi pada data baru dengan 3 input
print(
    "Hasil Prediksi:",
    clf.predict([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
)

# Menampilkan bentuk (shape) dari bobot pada setiap lapisan
print("Dimensi Bobot (Coefs Shape):", [coef.shape for coef in clf.coefs_])
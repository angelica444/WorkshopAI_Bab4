"""Example 4.2: Multiple Layer Perceptron (MLP) Classifier."""

from sklearn.neural_network import MLPClassifier

# Dataset logika OR (serupa dengan latihan sebelumnya)
X = [[0.0, 0.0], [1., 1.], [0., 1.], [1., 0.]]
y = [0, 1, 1, 1]

# Inisialisasi model MLP dengan 2 hidden layers berukuran (5, 2)
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)

# Melatih model menggunakan data
clf.fit(X, y)

# Menguji prediksi pada data baru
print("Hasil Prediksi:", clf.predict([[2.0, 2.0], [-1.0, -2.0]]))

# Menampilkan bentuk (shape) dari bobot pada setiap lapisan
print("Dimensi Bobot (Coefs Shape):", [coef.shape for coef in clf.coefs_])
"""Example 4.20: Forecast airline passenger counts with a SimpleRNN."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


def convert_to_matrix(data, step):
    inputs, targets = [], []
    for start in range(len(data) - step):
        inputs.append(data[start : start + step])
        targets.append(data[start + step])
    return np.asarray(inputs), np.asarray(targets)


def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    frame = pd.read_csv(url, usecols=[1])
    values = frame.to_numpy(dtype="float32")
    step = 4
    split = int(len(values) * 0.8)

    train_values = values[:split]
    test_values = np.concatenate((values[split - step :], values[split:]), axis=0)
    train_x, train_y = convert_to_matrix(train_values, step)
    test_x, test_y = convert_to_matrix(test_values, step)
    train_x = train_x.reshape((-1, 1, step))
    test_x = test_x.reshape((-1, 1, step))

    model = Sequential(
        [
            SimpleRNN(32, input_shape=(1, step), activation="relu"),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.summary()
    model.fit(train_x, train_y, epochs=10, batch_size=16, verbose=2)

    train_predictions = model.predict(train_x, verbose=0)
    test_predictions = model.predict(test_x, verbose=0)
    predictions = np.concatenate((train_predictions, test_predictions))
    print("Training loss:", model.evaluate(train_x, train_y, verbose=0))

    plt.plot(values, label="Data")
    plt.plot(np.arange(step, step + len(predictions)), predictions, label="Prediction")
    plt.axvline(split, color="red", label="80% split")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
import math
import random

import numpy as np
from keras import Input
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import initializers


def gen_sequential_model():
    model = Sequential([
        Input(4, name='input_layer'),
        Dense(16, activation='sigmoid', name='hidden_layer1',
              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)),
        Dense(1, activation='relu', name='output_layer',
              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
    ])

    model.summary()
    model.compile(optimizer='sgd', loss='mse')

    return model


def gen_linear_regression_dateset(numofsamples=500, w1=3, w2=5, w3=12, w4=20, b=10):
    np.random.seed(42)
    x_arr = []
    x_arr_cal = []
    for i in range(numofsamples):
        x_arr.append([random.random(), random.random(), random.random(), random.random()])
        x_arr_cal.append([x_arr[i][0], math.pow(x_arr[i][1], 2), math.pow(x_arr[i][2], 3), math.pow(x_arr[i][3], 4)])

    X = np.array(x_arr)
    coef = np.array([w1, w2, w3, w4])

    bias = b
    y = np.matmul(np.array(x_arr_cal), coef.transpose()) + bias
    return X, y


def plot_loss_curve(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def predict_new_sample(model, x, w1=3, w2=5, w3=12, w4=20, b=10):
    x = x.reshape(1, 4)
    y_pred = model.predict(x)[0][0]

    y_actual = w1 * x[0][0] + w2 * math.pow(x[0][1], 2) + w3 * math.pow(x[0][2], 3) + w4 * math.pow(x[0][3], 4) + b

    print("y actual value = ", y_actual)
    print("y predicted value = ", y_pred)


model = gen_sequential_model()
X, y = gen_linear_regression_dateset(numofsamples=300000)
history = model.fit(X, y, epochs=300, verbose=2, validation_split=0.2, batch_size=256)
plot_loss_curve(history)
print("train loss=", history.history['loss'][-1])
print("test loss=", history.history['val_loss'][-1])

predict_new_sample(model, np.array([0.6, 0.3, 0.5, 0.9]))

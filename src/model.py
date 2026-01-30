import numpy as np
from sklearn.linear_model import LinearRegression

def train_sklearn(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_scratch(x_train, y_train, lr, iterations):
    w, b = 0.0, 0.0
    n = len(x_train)

    for _ in range(iterations):
        y_hat = w * x_train + b
        error = y_hat - y_train
        dw = (2/n) * (error * x_train).sum()
        db = (2/n) * error.sum()
        w -= lr * dw
        b -= lr * db

    return w, b

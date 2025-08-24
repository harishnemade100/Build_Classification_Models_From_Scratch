import numpy as np
import math

def sigmoid_function(z):
  return 1 / (1 + np.exp(-z))

def predict(X, weight):
  z = np.dot(X, weight)
  return sigmoid_function(z)  


def cross_entropy(y_pred, y):
  epsilon = 1e-10  # small value to avoid log(0)
  return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))


def gradient_descent(X , y, weights, learning_rate, epochs):
  for i in range(epochs):
    y_pred = predict(X, weights)
    error = y_pred - y
    gradient = np.dot(X.T, error) / len(y)
    weights -= learning_rate * gradient
    if i % 100 == 0:
      print(f"Loss at epoch {i}: {cross_entropy(y, y_pred)}")
  return weights

X = np.array([[1, 98], [1, 101], [1, 105]])  # [bias term, temperature]
y = np.array([0, 1, 1])
weights = np.random.randn(2)

trained_weights = gradient_descent(X, y, weights, learning_rate=0.01, epochs=1000) 
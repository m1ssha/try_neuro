import numpy as np


def tanh(x):
    """Функция активации tanh."""
    return np.tanh(x)


def tanh_derivative(x):
    """Производная функции активации tanh."""
    return 1 - np.tanh(x) ** 2


def sigmoid(z):
    """Функция активации sigmoid."""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """Производная функции активации sigmoid."""
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)


def relu_derivative(a):
    return (a > 0).astype(float)


def softmax(z):
    # Для числовой стабильности вычитаем максимум по строкам
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
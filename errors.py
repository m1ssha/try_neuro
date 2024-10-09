import numpy as np

def kl_divergence(y_true, y_pred, epsilon=1e-10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return np.sum(y_true * np.log(y_true / y_pred), axis=1).mean()

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))


def mean_squared_error(y_true, y_pred):
    y_pred = np.clip(y_pred, -np.inf, np.inf)
    squared_diff = (y_true - y_pred) ** 2
    sum_squared_diff = np.sum(squared_diff, axis=1)
    mse = np.mean(sum_squared_diff)

    return mse


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-13
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1))


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
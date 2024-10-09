import numpy as np
from functions import tanh, tanh_derivative, sigmoid, sigmoid_derivative, softmax
from errors import kl_divergence, huber_loss, mean_squared_error, cross_entropy_loss, categorical_cross_entropy
from plot import precision_score, recall_score
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Выберите 'sigmoid' или 'tanh' для активационной функции.")
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []

        # Инициализация параметров для Adam Optimizer
        self.initialize_adam()

    def initialize_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Инициализация моментов для Adam Optimizer.
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

        self.m_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)

        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1  # (n_samples, hidden_size)
        self.A1 = self.activation(self.Z1)      # (n_samples, hidden_size)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # (n_samples, output_size)
        self.A2 = softmax(self.Z2)               # (n_samples, output_size)
        return self.A2

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]

        dZ2 = y_pred - y_true  # (n_samples, output_size)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)  # (n_samples, hidden_size)
        dZ1 = dA1 * self.activation_derivative(self.A1)  # (n_samples, hidden_size)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.iterations += 1
        self.update_adam(dW1, db1, dW2, db2)

    def update_adam(self, dW1, db1, dW2, db2):
        """
        Обновление параметров с использованием Adam Optimizer.
        """
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)

        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)

        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)

        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)

        m_W1_corr = self.m_W1 / (1 - self.beta1 ** self.iterations)
        v_W1_corr = self.v_W1 / (1 - self.beta2 ** self.iterations)
        m_b1_corr = self.m_b1 / (1 - self.beta1 ** self.iterations)
        v_b1_corr = self.v_b1 / (1 - self.beta2 ** self.iterations)

        m_W2_corr = self.m_W2 / (1 - self.beta1 ** self.iterations)
        v_W2_corr = self.v_W2 / (1 - self.beta2 ** self.iterations)
        m_b2_corr = self.m_b2 / (1 - self.beta1 ** self.iterations)
        v_b2_corr = self.v_b2 / (1 - self.beta2 ** self.iterations)

        self.W1 -= self.learning_rate * m_W1_corr / (np.sqrt(v_W1_corr) + self.epsilon)
        self.b1 -= self.learning_rate * m_b1_corr / (np.sqrt(v_b1_corr) + self.epsilon)
        self.W2 -= self.learning_rate * m_W2_corr / (np.sqrt(v_W2_corr) + self.epsilon)
        self.b2 -= self.learning_rate * m_b2_corr / (np.sqrt(v_b2_corr) + self.epsilon)

    def train(self, X, y, epochs, print_loss=False):
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = categorical_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y, axis=1)

            accuracy = np.mean(y_pred_classes == y_true_classes)
            precision_score_value = precision_score(y_true_classes, y_pred_classes, average='weighted')
            recall_score_value = recall_score(y_true_classes, y_pred_classes, average='weighted')

            self.accuracy_history.append(accuracy)
            self.precision_history.append(precision_score_value)
            self.recall_history.append(recall_score_value)

            self.backward(X, y, y_pred)

            if print_loss and epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision_score_value:.4f}, Recall: {recall_score_value:.4f}')

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def plot_metrics(self):
        """Построение графиков метрик."""
        epochs = range(1, len(self.loss_history) + 1)

        plt.figure(figsize=(12, 8))

        # График потерь
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.loss_history, label='Loss', color='blue')
        plt.xlabel('Эпохи')
        plt.ylabel('Loss')
        plt.title('Потери')
        plt.grid(True)
        plt.legend()

        # График точности
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.accuracy_history, label='Accuracy', color='green')
        plt.xlabel('Эпохи')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.grid(True)
        plt.legend()

        # График Precision
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.precision_history, label='Precision', color='red')
        plt.xlabel('Эпохи')
        plt.ylabel('Precision')
        plt.title('Precision')
        plt.grid(True)
        plt.legend()

        # График Recall
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.recall_history, label='Recall', color='orange')
        plt.xlabel('Эпохи')
        plt.ylabel('Recall')
        plt.title('Recall')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


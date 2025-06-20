import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Visualisation comparative
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 5))

plt.subplot(141)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(142)
plt.plot(x, relu(x), label='ReLU')
plt.title('ReLU')
plt.grid(True)

plt.subplot(143)
plt.plot(x, tanh(x), label='Tanh')
plt.title('Tanh')
plt.grid(True)

# Exemple Softmax
x_softmax = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
plt.subplot(144)
plt.bar(range(len(x_softmax)), softmax(x_softmax))
plt.title('Softmax')

plt.tight_layout()
plt.show()
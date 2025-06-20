import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Visualisation
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Fonction Sigmoïde')
plt.xlabel('Entrée')
plt.ylabel('Sortie')
plt.grid(True)
plt.show()


def relu(x):
    return np.maximum(0, x)

# Visualisation
x = np.linspace(-10, 10, 100)
y = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Fonction ReLU')
plt.xlabel('Entrée')
plt.ylabel('Sortie')
plt.grid(True)
plt.show()

def tanh(x):
    return np.tanh(x)

# Visualisation
x = np.linspace(-10, 10, 100)
y = tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Fonction Tangente Hyperbolique')
plt.xlabel('Entrée')
plt.ylabel('Sortie')
plt.grid(True)
plt.show()
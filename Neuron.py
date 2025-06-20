import numpy as np

def simple_neuron(inputs, weights, bias):
    # Calculate the weighted sum of inputs
    total = np.dot(inputs, weights) + bias
    
    # Activation function (we'll use a simple step function)
    output = 1 if total > 0 else 0
    
    return output

# Example inputs
inputs = np.array([2, 3, 1])
weights = np.array([0.5, -0.8, 0.3])
bias = 0  # Bias term

result = simple_neuron(inputs, weights, bias)
print("Neuron output:", result)

def better_neuron(inputs, weights, bias):
    # Calculate the weighted sum
    total = np.dot(inputs, weights) + bias
    
    # Sigmoid activation function (more nuanced)
    output = 1 / (1 + np.exp(-total))
    
    return output

# Try different inputs and weights
inputs = np.array([2, 3, 1])
weights = np.array([0.5, -0.8, 0.3])
bias = -1

result = better_neuron(inputs, weights, bias)
print("Neuron output:", result)


import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        # Initialisation aléatoire des poids
        self.weights = np.random.randn(3, 1)
        self.bias = np.random.randn(1)
        print("self weithts" , self.weights, "self bias" ,self.bias)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, inputs):
        # Calculer la somme pondérée
        total = np.dot(inputs, self.weights) + self.bias
        
        # Utiliser la fonction sigmoid
        output_sigmoid = self.sigmoid(total)
        
        # Utiliser la fonction ReLU
        output_relu = self.relu(total)
        
        return {
            'sigmoid_output': output_sigmoid,
            'relu_output': output_relu
        }

# Tester le réseau
network = SimpleNeuralNetwork()
inputs = np.array([2, 3, 4])
results = network.forward(inputs)

print("Résultats avec différentes fonctions d'activation:")
print("Sigmoid:", results['sigmoid_output'])
print("ReLU:", results['relu_output'])
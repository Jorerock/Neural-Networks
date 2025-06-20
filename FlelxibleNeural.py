import numpy as np
import matplotlib.pyplot as plt

class FlexibleNeuralNetwork:
    def __init__(self, activation):
        self.activation = activation
        
    def initialize_weights(self, input_size):
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.random.randn(1)
        # print("Poids initiaux:", self.weights, "Biais initial:", self.bias)
    

    def choose_activation(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Activation non supportée")
    
    def forward(self, inputs):
        if self.weights is None or self.bias is None:
            self.initialize_weights(len(inputs))
        # Calcul de la somme pondérée
        Sommepondere = np.dot(inputs, self.weights) + self.bias

        output = self.choose_activation(Sommepondere)
        return output
        
        

# Essayez avec différentes activations
network1 = FlexibleNeuralNetwork(activation='sigmoid')
inputs = np.array([0.8, 0.5, 1])
print("Entrees:", inputs)
network1.initialize_weights(len(inputs))

# Premier réseau de neurones
layer = [None] * 4
for i in range(4):
    # Réinitialiser les poids et le biais pour chaque itération
    network1.initialize_weights(len(inputs))
    
    # Calculer la sortie
    sortie = network1.forward(inputs)
    print("layer 1, neuronne", i, ":", sortie)
    layer[i] = network1.forward(inputs).item()

print(layer)

# deuxième réseau de neurones
network2 = FlexibleNeuralNetwork(activation='relu')
inputs = layer
layer2 = [None] * 4
for i in range(4):
    # Réinitialiser les poids et le biais pour chaque itération
    network2.initialize_weights(len(inputs))
    
    # Calculer la sortie
    sortie = network2.forward(inputs)
    print("layer 2, neuronne", i, ":", sortie)
    layer2[i] = network2.forward(inputs).item()

print(layer2)

# Visualisation des sorties des couches
plt.figure(figsize=(10, 6))     

for i in range(4):
    plt.plot(layer[i], label=f'Layer {i+1} ({network1.activation})')
plt.title('Sorties des couches avec activation ' + network1.activation)
plt.xlabel('Index de la couche')
plt.ylabel('Valeur de sortie')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from ImportImage import import_image


class FlexibleNeuralNetwork:
    def __init__(self, activation , number_0f_neurons):
        self.activation = activation
        self.number_0f_neurons = number_0f_neurons
        
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
        
        


# # Classe pour créer des couches de neurones flexibles
# class layers:
#     def __init__(self, activation, number_0f_neurons):
#         self.activation = activation
#         self.number_0f_neurons = number_0f_neurons
#         self.network = FlexibleNeuralNetwork(activation, number_0f_neurons)
  
#     def Result(self, inputs):
#         for i in range(self.number_0f_neurons):
#             # Réinitialiser les poids et le biais pour chaque itération
#             self.network.initialize_weights(len(inputs))
            
#             # Calculer la sortie
#             sortie = self.network.forward(inputs)
#             # print("layer 1, neuronne", i, ":", sortie)
#             layer[i] = self.network.forward(inputs).item()

#         return layer
    
    





# Importer l'image et la normaliser
import_image = import_image()
inputs = import_image.import_image_npy("Numbers/number.npy")
# inputs = import_image.import_image_png("Numbers/number.png")
inputs = inputs.flatten()  # Aplatir l'image en un vecteur
inputs = inputs / 255.0  # Normaliser les valeurs des pixels entre 0 et 1

# Afficher les entrées
print("Entrées normalisées:", inputs)
network1 = FlexibleNeuralNetwork(activation='sigmoid',number_0f_neurons=128)

# Premier réseau de neurones
layer = [None] * network1.number_0f_neurons
for i in range(network1.number_0f_neurons):
    # Réinitialiser les poids et le biais pour chaque itération
    network1.initialize_weights(len(inputs))
    
    # Calculer la sortie
    sortie = network1.forward(inputs)
    # print("layer 1, neuronne", i, ":", sortie)
    layer[i] = network1.forward(inputs).item()

# print(layer)

# deuxième réseau de neurones
network2 = FlexibleNeuralNetwork(activation='relu',number_0f_neurons=64)
inputs = layer
layer2 = [None] * network2.number_0f_neurons
for i in range(network2.number_0f_neurons):
    # Réinitialiser les poids et le biais pour chaque itération
    network2.initialize_weights(len(inputs))
    
    # Calculer la sortie
    sortie = network2.forward(inputs)
    # print("layer 2, neuronne", i, ":", sortie)
    layer2[i] = network2.forward(inputs).item()

# print(layer2)

# Sortie
network3 = FlexibleNeuralNetwork(activation='relu',number_0f_neurons=10)
inputs = layer2
layer3 = [None] * network3.number_0f_neurons
for i in range(network3.number_0f_neurons):
    # Réinitialiser les poids et le biais pour chaque itération
    network3.initialize_weights(len(inputs))
    
    # Calculer la sortie
    sortie = network3.forward(inputs)
    # print("layer 3, neuronne", i, ":", sortie)
    layer3[i] = network3.forward(inputs).item()
# print(layer3)


# Affichage des sorties finales
print("Sorties finales des couches:")
print("Layer 1:", layer)
print("Layer 2:", layer2)
print("Layer 3:", layer3)


# # Visualisation des sorties des couches
# plt.figure(figsize=(10, 6))     

# for i in range(4):
#     plt.plot(layer[i], label=f'Layer {i+1} ({network1.activation})')
# plt.title('Sorties des couches avec activation ' + network1.activation)
# plt.xlabel('Index de la couche')
# plt.ylabel('Valeur de sortie')
# plt.legend()
# plt.grid(True)
# plt.show()

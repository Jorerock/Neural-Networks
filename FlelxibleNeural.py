import numpy as np
import matplotlib.pyplot as plt
from ImportImage import import_image


class Neuronne:
    def __init__(self, activation , number_0f_neurons):
        self.activation = activation
        self.number_0f_neurons = number_0f_neurons
        self.weights = None
        self.bias = None
        print("Activation:", self.activation, "Nombre de neurones:", self.number_0f_neurons)
        
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
        elif self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x))
            return exp_x / exp_x.sum(axis=0, keepdims=True)
        else:
            raise ValueError("Activation non supportée")
    
    def forward(self, inputs):
        if self.weights is None or self.bias is None:
            if self.weights is None:
                self.initialize_weights(len(inputs))
        Sommepondere = np.dot(inputs, self.weights) + self.bias

        output = self.choose_activation(Sommepondere)
        return output
        
        


# Classe pour créer des couches de neurones flexibles
class layers:
    def __init__(self, activation, number_0f_neurons):
        self.activation = activation
        self.number_0f_neurons = number_0f_neurons
        self.network = Neuronne(activation, number_0f_neurons)
  
    def Result(self, inputs):
        result = [None] * self.number_0f_neurons
        for i in range(self.number_0f_neurons):
            self.network.initialize_weights(len(inputs))
            result[i] = self.network.forward(inputs).item()
        return result


# Classe pour créer un réseau de neurones flexible
class NeuralNetwork:
    def __init__(self, numberofLayer,Layer_neurons_and_activation = [('sigmoid', 128), ('relu', 64), ('relu', 10)]):
        self.numberofLayer = numberofLayer
        self.Layer_neurons_and_activation = Layer_neurons_and_activation
        self.layers = [None] * numberofLayer
        for i in range(numberofLayer):
            activation, number_0f_neurons = Layer_neurons_and_activation[i]
            self.layers[i] = layers(activation, number_0f_neurons)

    
    def Go(self, inputs):
        for i in range(self.numberofLayer):
            inputs = self.layers[i].Result(inputs)
        #une fonction softmax pour la dernière couche de sortie
        exp_x = np.exp(inputs - np.max(inputs))
        return exp_x / exp_x.sum(axis=0, keepdims=True)





# Importer l'image et la normaliser
import_image = import_image()
inputs = import_image.import_image_npy("Numbers/number.npy")
# inputs = import_image.import_image_png("Numbers/number.png")
inputs = inputs.flatten()  # Aplatir l'image en un vecteur
inputs = inputs / 255.0  # Normaliser les valeurs des pixels entre 0 et 1



NeuralNetwork1 = NeuralNetwork(numberofLayer=3, Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('relu', 10)])

for i in range(3):
    resultat = (NeuralNetwork1.Go(inputs))
    print(resultat)
    for i in range(resultat.shape[0]):
        print(i+1, ":", resultat[i]/100)













#trash place:

# # Afficher les entrées
# print("Entrées normalisées:", inputs)
# layer1 = layers(activation='sigmoid', number_0f_neurons=128)
# inputs = layer1.Result(inputs)  

# # Premier réseau de neurones
# # network1 = Neuronne(activation='sigmoid',number_0f_neurons=128)
# # layer = [None] * network1.number_0f_neurons
# # for i in range(network1.number_0f_neurons):
# #     # Réinitialiser les poids et le biais pour chaque itération
# #     network1.initialize_weights(len(inputs))
    
# #     # Calculer la sortie
# #     sortie = network1.forward(inputs)
# #     # print("layer 1, neuronne", i, ":", sortie)
# #     layer[i] = network1.forward(inputs).item()

# # print(layer)
# # deuxième réseau de neurones
# network2 = Neuronne(activation='relu',number_0f_neurons=64)
# # inputs = layerResult
# layer2 = [None] * network2.number_0f_neurons
# for i in range(network2.number_0f_neurons):
#     # Réinitialiser les poids et le biais pour chaque itération
#     network2.initialize_weights(len(inputs))
    
#     # Calculer la sortie
#     sortie = network2.forward(inputs)
#     # print("layer 2, neuronne", i, ":", sortie)
#     layer2[i] = network2.forward(inputs).item()

# # print(layer2)

# # Sortie
# network3 = Neuronne(activation='relu',number_0f_neurons=10)
# inputs = layer2
# layer3 = [None] * network3.number_0f_neurons
# for i in range(network3.number_0f_neurons):
#     # Réinitialiser les poids et le biais pour chaque itération
#     network3.initialize_weights(len(inputs))
    
#     # Calculer la sortie
#     sortie = network3.forward(inputs)
#     # print("layer 3, neuronne", i, ":", sortie)
#     layer3[i] = network3.forward(inputs).item()
# # print(layer3)


# # Affichage des sorties finales
# print("Sorties finales des couches:")

# print("Layer 2:", layer2)
# print("Layer 3:", layer3)


# # # Visualisation des sorties des couches
# # plt.figure(figsize=(10, 6))     

# # for i in range(4):
# #     plt.plot(layer[i], label=f'Layer {i+1} ({network1.activation})')
# # plt.title('Sorties des couches avec activation ' + network1.activation)
# # plt.xlabel('Index de la couche')
# # plt.ylabel('Valeur de sortie')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

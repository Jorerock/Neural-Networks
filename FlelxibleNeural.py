import numpy as np
import matplotlib.pyplot as plt
from ImportImage import import_image




class Neuronne:
    def __init__(self, activation):
        self.activation = activation

    def choose_activation(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        # elif self.activation == 'softmax':
        #     exp_x = np.exp(x - np.max(x))
        #     return exp_x / exp_x.sum(axis=0, keepdims=True)
        else:
            raise ValueError("Activation non supportée")
    
    def forward(self, inputs,weight,bias):
        Sommepondere = np.dot(inputs, weight) + bias #produit des Vecteurs de matrices
        output = self.choose_activation(Sommepondere)
        return output

class layers:
    def __init__(self, activation, number_of_neurons):
        self.activation = activation
        self.number_of_neurons = number_of_neurons
        self.network = Neuronne(activation)

    def Result(self, inputs,weights, bias):
        result = [] 
        for i in range(self.number_of_neurons):
            result.append(self.network.forward(inputs,weights[:, i] ,bias[i]))
        return np.array(result)

class NeuralNetwork:
    def __init__(self,inputs_size = 784  ,numberofLayer = 3 ,Layer_neurons_and_activation = [('sigmoid', 128), ('relu', 64), ('relu', 10)]):
        self.numberofLayer = numberofLayer
        self.Layer_neurons_and_activation = Layer_neurons_and_activation
        self.weights = []
        self.biases = []
        self.Initialise_weights_and_bias(inputs_size)
        

    def Initialise_weights_and_bias(self,inputs_size):
        prev_size = inputs_size
        for layer in range(self.numberofLayer):
            activation, num_neurons = self.Layer_neurons_and_activation[layer]
            weights = np.random.randn(prev_size, num_neurons) * 0.1
            bias = np.zeros(num_neurons)
            self.weights.append(weights)
            self.biases.append(bias)
            prev_size = num_neurons
        print("Poids et biais initialisés:")
        for i in range(self.numberofLayer):
            print(f"Couche {i+1}: weights {self.weights[i].shape}, bias {self.biases[i].shape}")
    
    def Go(self, inputs):
        current_inputs = inputs
        for i in range(self.numberofLayer):
            layer = layers(
            activation=self.Layer_neurons_and_activation[i][0], 
            number_of_neurons=self.Layer_neurons_and_activation[i][1]
            )
            current_inputs = layer.Result(current_inputs, self.weights[i], self.biases[i])    

            # inputs = self.layers[i].Result(inputs)
        #une fonction softmax pour la dernière couche de sortie

        # exp_x = np.exp(inputs - np.max(inputs))
        # return exp_x / exp_x.sum(axis=0, keepdims=True)
        return current_inputs




# Importer l'image et la normaliser
import_image = import_image()
inputs = import_image.import_image_npy("Numbers/number.npy")
# inputs = import_image.import_image_png("Numbers/number.png")
inputs = inputs.flatten()  # Aplatir l'image en un vecteur
inputs = inputs / 255.0  # Normaliser les valeurs des pixels entre 0 et 1


print("Parametrage reseau:")
NeuralNetwork1 = NeuralNetwork(inputs_size = 784, numberofLayer=3, Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('relu', 10)])
# sbiais et poids  784*128 +128 *64 + 64*10 et 128+63+10 biases

print("\nGo:",)
y_true = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # 4 est la reponse juste
# loss = -np.sum(true * np.log(resultat + 1e-9))
Loss = []
for i in range(3):
    resultat = (NeuralNetwork1.Go(inputs))
    print(resultat)
    for i in range(resultat.shape[0]):
        print(f"{i}: {resultat[i]*100:.2f}%")
        Loss(resultat,y_true)


#dit cross entropy
def Loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9))
    



# # for i in range(4):
# #     plt.plot(layer[i], label=f'Layer {i+1} ({network1.activation})')
# # plt.title('Sorties des couches avec activation ' + network1.activation)
# # plt.xlabel('Index de la couche')
# # plt.ylabel('Valeur de sortie')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

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
        else:
            raise ValueError("Activation non supportée")
    
    def forward(self, inputs,weight,bias):
        Sommepondere = np.dot(inputs, weight) + bias #produit des Vecteurs de matrices
        output = self.choose_activation(Sommepondere)
        return output ,Sommepondere 

class layers:
    def __init__(self, activation, number_of_neurons):
        self.activation = activation
        self.number_of_neurons = number_of_neurons
        self.network = Neuronne(activation)

    def Result(self, inputs,weights, bias):
        result = [] 
        z_values = [] #Matrice des mes Sommes ponderes
        for i in range(self.number_of_neurons):
            output , Sommepondere  = self.network.forward(inputs,weights[:, i] ,bias[i])
            result.append(output)
            z_values.append(Sommepondere)
        return np.array(result),np.array(z_values)

class NeuralNetwork:
    def __init__(self,inputs_size = 784  ,numberofLayer = 3 ,Layer_neurons_and_activation = [('sigmoid', 128), ('relu', 64), ('relu', 10)]):
        self.numberofLayer = numberofLayer
        self.Layer_neurons_and_activation = Layer_neurons_and_activation
        self.weights = []
        self.biases = []
        self.Initialise_weights_and_bias(inputs_size)
        self.inputs= []
        self.z_values = []
        self.output= []
        

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
        self.inputs= []
        self.z_values = []
        self.output= []
        current_inputs = inputs
        for i in range(self.numberofLayer):
            layer = layers(
            activation=self.Layer_neurons_and_activation[i][0], 
            number_of_neurons=self.Layer_neurons_and_activation[i][1]
            )
            self.inputs.append(current_inputs)
            current_inputs , z_values = layer.Result(current_inputs, self.weights[i], self.biases[i])
            self.z_values.append(z_values)
            self.output.append(current_inputs)
   

        derniere_sortie = self.output[-1]  # Dernière couche
        gradient_poids = np.outer(self.inputs[-1], Graditant(derniere_sortie, y_true) * relu_derivative(self.z_values[-1]))
        print("Gradient des poids : ",gradient_poids)
        gradient_biais = Graditant(derniere_sortie, y_true) * relu_derivative(self.z_values[-1])
        print("Gradient des biais  : ",gradient_biais)

        return current_inputs





# Importer l'image et la normaliser
import_image = import_image()
inputs = import_image.import_image_npy("Numbers/number.npy")
# inputs = import_image.import_image_png("Numbers/number.png")
inputs = inputs.flatten()  # Aplatir l'image en un vecteur
inputs = inputs / 255.0  # Normaliser les valeurs des pixels entre 0 et 1


#dit 
def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def Graditant(y_pred, y_true):
    return y_pred-y_true


def relu_derivative(z):
    return np.where(z > 0, 1, 0) #si z > 0 alors dérivée = 1, sinon dérivée = 0 pour chaque valeur de mon tableau

# def Deriver_Partiel(Weights,valeur,Bias,Somepondere,function, Coss):
print("Parametrage reseau:")
NeuralNetwork1 = NeuralNetwork(inputs_size = 784, numberofLayer=3, Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('relu', 10)])
# sbiais et poids  784*128 +128 *64 + 64*10 poids et 128+63+10 biases
print("\nGo:",)
y_true = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # 4 est la reponse juste pour l'image actuelle

for i in range(1):
    resultat = (NeuralNetwork1.Go(inputs))
    print(resultat)
    for i in range(resultat.shape[0]):
        print(f"{i}: {resultat[i]*100:.2f}%")



# # for i in range(4):
# #     plt.plot(layer[i], label=f'Layer {i+1} ({network1.activation})')
# # plt.title('Sorties des couches avec activation ' + network1.activation)
# # plt.xlabel('Index de la couche')
# # plt.ylabel('Valeur de sortie')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

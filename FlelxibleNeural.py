import numpy as np
import matplotlib.pyplot as plt
import pickle
from ImportImage import import_image
import tensorflow as tf #ici tensorflow est seulement utiliser pour importe la bibliotheque MNIST (content une enorme quantite de nombre ecrit a la main pour entrainer mon model)



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
        self.gradients_poids = []  # Liste de matrices
        self.gradients_biais = []  # Liste de vecteurs
        self.deltas = []
        self.learning_rate = 0.1 #vitesse d'apprentissage du reseau

    def Initialise_weights_and_bias(self,inputs_size):
        prev_size = inputs_size
        for layer in range(self.numberofLayer):
            activation, num_neurons = self.Layer_neurons_and_activation[layer]
            weights = np.random.randn(prev_size, num_neurons) * 0.1
            bias = np.zeros(num_neurons)
            self.weights.append(weights)
            self.biases.append(bias)
            prev_size = num_neurons
        # print("Poids et biais initialisés:")
        # for i in range(self.numberofLayer):
            # print(f"Couche {i+1}: weights {self.weights[i].shape}, bias {self.biases[i].shape}")

    def Gradiant_layer(self,layer_Number,activation,delta_from_next_layer):
        if(activation == "relu"):
            gradient_poids = np.outer(self.inputs[layer_Number], delta_from_next_layer * relu_derivative(self.z_values[layer_Number]))
            # print("layer : ",layer_Number ,"Gradient des poids : ",gradient_poids)
            gradient_biais = delta_from_next_layer * relu_derivative(self.z_values[layer_Number])
            # print("Gradient des biais  : ",gradient_biais)
        elif (activation == "sigmoid"):
            gradient_poids = np.outer(self.inputs[layer_Number], delta_from_next_layer * sigmoid_derivative(self.z_values[layer_Number]))
            # print("Gradient des poids : ",gradient_poids)
            gradient_biais = delta_from_next_layer * sigmoid_derivative(self.z_values[layer_Number])
            # print("Gradient des biais  : ",gradient_biais)
        return gradient_poids , gradient_biais
    

    def Calc_Delta(self, y_true):
        self.deltas = []
        #delta de la derniere couche
        derniere_sortie = self.output[-1]  # Dernière couche
        activation_derniere = self.Layer_neurons_and_activation[-1][0]
        if activation_derniere == "relu":
            delta = Delta(derniere_sortie, y_true) * relu_derivative(self.z_values[-1])
        elif activation_derniere == "sigmoid":
            delta = Delta(derniere_sortie, y_true) * sigmoid_derivative(self.z_values[-1])
        self.deltas.append(delta)
        #on remonte les couches
        for layer in range(self.numberofLayer-2, -1, -1):  
            if(self.Layer_neurons_and_activation[layer][0] == "relu"):
                delta_precedent = delta.dot(self.weights[layer+1].T) * relu_derivative(self.z_values[layer])
                self.deltas.insert(0, delta_precedent) 
            elif (self.Layer_neurons_and_activation[layer][0] == "sigmoid"):
                delta_precedent = delta.dot(self.weights[layer+1].T) * sigmoid_derivative(self.z_values[layer])
                self.deltas.insert(0, delta_precedent) 
            delta = delta_precedent 


    def save_model(self, filepath):
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'numberofLayer': self.numberofLayer,
            'Layer_neurons_and_activation': self.Layer_neurons_and_activation,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modèle sauvegardé dans {filepath}")

    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data['weights']
            self.biases = model_data['biases']
            self.numberofLayer = model_data['numberofLayer']
            self.Layer_neurons_and_activation = model_data['Layer_neurons_and_activation']
            self.learning_rate = model_data['learning_rate']
            print(f"Modèle chargé depuis {filepath}")
            return True
        except FileNotFoundError:
            print(f"Fichier {filepath} non trouvé. Initialisation avec des poids aléatoires.")
            return False
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            return False

    
    def Go(self, inputs):
        print("go")
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
   
        self.Calc_Delta(y_true=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.gradients_poids = []
        self.gradients_biais = []
        # Calculer tous les gradients en utilisant les deltas
        for layer in range(self.numberofLayer):
            activation = self.Layer_neurons_and_activation[layer][0]
            grad_w, grad_b = self.Gradiant_layer(layer, activation, self.deltas[layer])
            self.gradients_poids.append(grad_w)
            self.gradients_biais.append(grad_b)
        
        for i in range(self.numberofLayer):
            self.weights[i] -= self.learning_rate * self.gradients_poids[i]
            self.biases[i] -= self.learning_rate * self.gradients_biais[i]
        # print(self.gradients_biais)
        return current_inputs
    
    def training(self, batch_x,batch_y):
        self.inputs= []
        self.z_values = []
        self.output= []
        current_inputs = batch_x
        for i in range(self.numberofLayer):
            layer = layers(
            activation=self.Layer_neurons_and_activation[i][0], 
            number_of_neurons=self.Layer_neurons_and_activation[i][1]
            )
            self.inputs.append(current_inputs)
            current_inputs , z_values = layer.Result(current_inputs, self.weights[i], self.biases[i])
            self.z_values.append(z_values)
            self.output.append(current_inputs)
   
        self.Calc_Delta(y_true=batch_y)
        self.gradients_poids = []
        self.gradients_biais = []
        # Calculer tous les gradients en utilisant les deltas
        for layer in range(self.numberofLayer):
            activation = self.Layer_neurons_and_activation[layer][0]
            grad_w, grad_b = self.Gradiant_layer(layer, activation, self.deltas[layer])
            self.gradients_poids.append(grad_w)
            self.gradients_biais.append(grad_b)
        
        for i in range(self.numberofLayer):
            self.weights[i] -= self.learning_rate * self.gradients_poids[i]
            self.biases[i] -= self.learning_rate * self.gradients_biais[i]
        # print(self.gradients_biais)
        return current_inputs    

#dit 
def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9))

def Delta(y_pred, y_true):
    return y_pred-y_true


def relu_derivative(z):
    return np.where(z > 0, 1, 0) #si z > 0 alors dérivée = 1, sinon dérivée = 0 pour chaque valeur de mon tableau

def sigmoid_derivative(z):
    sigmoid_z = 1 / (1 + np.exp(-z))
    return sigmoid_z * (1 - sigmoid_z)


#*******************************************************************************************************************************************************************************************

# Importer l'image et la normaliser
# import_image = import_image()
# inputs = import_image.import_image_npy("Numbers/number.npy")
# # inputs = import_image.import_image_png("Numbers/number.png")
# inputs = inputs.flatten()  # Aplatir l'image en un vecteur
# inputs = inputs / 255.0  # Normaliser les valeurs des pixels entre 0 et 1

# # def Deriver_Partiel(Weights,valeur,Bias,Somepondere,function, Coss):
# print("Parametrage reseau:")
# NeuralNetwork1 = NeuralNetwork(inputs_size = 784, numberofLayer=3, Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('relu', 10)])
# # sbiais et poids  784*128 +128 *64 + 64*10 poids et 128+63+10 biases
# print("\nGo:",)

# for i in range(1):
#     resultat = (NeuralNetwork1.Go(inputs))
#     # print(resultat)
#     for i in range(resultat.shape[0]):
#         print(f"{i}: {resultat[i]*100:.2f}%")

# Créer et entraîner votre réseau
# NeuralNetwork1 = NeuralNetwork(inputs_size=784, numberofLayer=3, Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('relu', 10)])

# # Essayer de charger un modèle existant
# if not NeuralNetwork1.load_model("mon_modele.pkl"):
#     print("Nouveau modèle créé")

# # Entraîner votre réseau
# for epoch in range(1000):
#     resultat = NeuralNetwork1.Go(inputs)
#     for i in range(resultat.shape[0]):
#         print(f"{i}: {resultat[i]*100:.2f}%")

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch} terminé")

# # Sauvegarder après l'entraînement
# NeuralNetwork1.save_model("mon_modele.pkl")




def load_mnist_data():
    """Charge les données MNIST"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normaliser les pixels entre 0 et 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Aplatir les images (28x28 -> 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    # Convertir les labels en one-hot encoding
    y_train_onehot = np.zeros((y_train.shape[0], 10))
    y_test_onehot = np.zeros((y_test.shape[0], 10))
    
    for i in range(y_train.shape[0]):
        y_train_onehot[i, y_train[i]] = 1
    for i in range(y_test.shape[0]):
        y_test_onehot[i, y_test[i]] = 1
    
    return (x_train, y_train_onehot), (x_test, y_test_onehot)

# Charger les données
(x_train, y_train), (x_test, y_test) = load_mnist_data()
print(f"Données d'entraînement: {x_train.shape}")
print(f"Labels d'entraînement: {y_train.shape}")

(x_train, y_train), (x_test, y_test) = load_mnist_data()

# Créer votre réseau
NeuralNetwork1 = NeuralNetwork(inputs_size=784, numberofLayer=3, 
                              Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('relu', 10)])

# Essayer de charger un modèle existant
if not NeuralNetwork1.load_model("mnist_model_final.pkl"):
    print("Nouveau modèle créé")

# Entraînement
epochs = 3
batch_size = 32

print("Début de l'entraînement...")
for epoch in range(epochs):
    # Mélanger les données
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    total_loss = 0
    correct_predictions = 0
    
    # Entraînement par batch
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train_shuffled[i:i+batch_size]
        batch_y = y_train_shuffled[i:i+batch_size]
        
        for j in range(len(batch_x)):
            result = NeuralNetwork1.training(batch_x[j], batch_y[j])
            
            predicted = np.argmax(result)
            actual = np.argmax(batch_y[j])
            if predicted == actual:
                correct_predictions += 1
    

    accuracy = correct_predictions / len(x_train) * 100
    print(f"Epoch {epoch}: Précision = {accuracy:.2f}%")
    # Afficher les stats toutes les 50 époques
    if epoch % 50 == 0:
        accuracy = correct_predictions / len(x_train) * 100
        print(f"Epoch {epoch}: Précision = {accuracy:.2f}%")
    
    # Sauvegarder le modèle toutes les 100 époques
    if epoch % 100 == 0 and epoch > 0:
        NeuralNetwork1.save_model(f"mnist_model_epoch_{epoch}.pkl")

# Sauvegarder le modèle final
NeuralNetwork1.save_model("mnist_model_final.pkl")
print("Entraînement terminé!")
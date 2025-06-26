import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import numpy as np
from ImportImage import import_image
# Importez vos autres modules ici

import scipy.ndimage

def center_image(img_array):
    """
    Centre une image 28x28 (normalisée entre 0 et 1).
    :param img_array: np.array 28x28
    :return: image recentrée 28x28
    """
    if img_array.shape != (28, 28):
        raise ValueError("Image doit être 28x28")

    # Trouver les coordonnées où il y a du "noir"
    rows = np.any(img_array > 0.1, axis=1)
    cols = np.any(img_array > 0.1, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img_array  # Image vide, pas besoin de centrer

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    cropped = img_array[top:bottom+1, left:right+1]

    # Zoom proportionnel pour qu'il tienne dans 20x20
    new_size = max(cropped.shape)
    if new_size > 0:
        scale = 20.0 / new_size
        resized = scipy.ndimage.zoom(cropped, zoom=scale, order=1)
    else:
        resized = cropped

    # Créer une image vide 28x28
    new_img = np.zeros((28, 28))

    h, w = resized.shape
    top_offset = (28 - h) // 2
    left_offset = (28 - w) // 2

    new_img[top_offset:top_offset+h, left_offset:left_offset+w] = resized

    return new_img
        
class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Interface")
        self.root.geometry("600x500")
        
        # Variables
        self.network = None
        self.mnist_data = None
        self.current_image = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # Titre principal
        title_label = tk.Label(self.root, text="🧠 Neural Network Interface", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Section 1: Gestion du modèle
        model_frame = ttk.LabelFrame(main_frame, text="Gestion du Modèle")
        model_frame.pack(fill="x", pady=5)
        
        ttk.Button(model_frame, text="📦 Créer Nouveau Modèle", 
                  command=self.create_model).pack(side="left", padx=5, pady=5)
        ttk.Button(model_frame, text="📂 Charger Modèle", 
                  command=self.load_model).pack(side="left", padx=5, pady=5)
        ttk.Button(model_frame, text="💾 Sauvegarder Modèle", 
                  command=self.save_model).pack(side="left", padx=5, pady=5)
        
        # Section 2: Données
        data_frame = ttk.LabelFrame(main_frame, text="Gestion des Données")
        data_frame.pack(fill="x", pady=5)
        
        ttk.Button(data_frame, text="📥 Charger MNIST", 
                  command=self.load_mnist).pack(side="left", padx=5, pady=5)
        ttk.Button(data_frame, text="🖼️ Charger Image", 
                  command=self.load_image).pack(side="left", padx=5, pady=5)
        ttk.Button(data_frame, text="🖌️ Cree Image", 
                command=self.Create_image).pack(side="left", padx=5, pady=5)
        
        
        # Section 3: Entraînement
        train_frame = ttk.LabelFrame(main_frame, text="Entraînement")
        train_frame.pack(fill="x", pady=5)
        
        # Paramètres d'entraînement
        param_frame = ttk.Frame(train_frame)
        param_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Époques:").pack(side="left")
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).pack(side="left", padx=5)
        
        ttk.Label(param_frame, text="Learning Rate:").pack(side="left", padx=(20,0))
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.lr_var, width=10).pack(side="left", padx=5)
        
        # Boutons d'entraînement
        train_buttons = ttk.Frame(train_frame)
        train_buttons.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(train_buttons, text="🚀 Entraîner (Simple)", 
                  command=self.train_simple).pack(side="left", padx=5)
        ttk.Button(train_buttons, text="📊 Entraîner (MNIST)", 
                  command=self.train_mnist).pack(side="left", padx=5)
        ttk.Button(train_buttons, text="⏹️ Arrêter", 
                  command=self.stop_training).pack(side="left", padx=5)
        
        # Section 4: Test et Prédiction
        test_frame = ttk.LabelFrame(main_frame, text="Test et Prédiction")
        test_frame.pack(fill="x", pady=5)
        
        ttk.Button(test_frame, text="🎯 Prédire Image Courante", 
                  command=self.predict_current).pack(side="left", padx=5, pady=5)
        ttk.Button(test_frame, text="📈 Tester sur MNIST", 
                  command=self.test_mnist).pack(side="left", padx=5, pady=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Logs")
        log_frame.pack(fill="both", expand=True, pady=5)
        
        # Zone de texte avec scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(text_frame, height=10, state="disabled")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Variables de contrôle
        self.training_active = False
        
        # Log initial
        self.log("🎉 Interface initialisée. Prêt à utiliser!")
    
    def log(self, message):
        """Ajoute un message dans les logs"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update()
    
    def create_model(self):
        """Crée un nouveau modèle"""
        try:
            from FlelxibleNeural import NeuralNetwork  # Remplacez par votre import
            self.network = NeuralNetwork(
                inputs_size=784, 
                numberofLayer=3, 
                Layer_neurons_and_activation=[('sigmoid', 128), ('relu', 64), ('sigmoid', 10)]
            )
            self.log("✅ Nouveau modèle créé avec succès!")
        except Exception as e:
            self.log(f"❌ Erreur création modèle: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la création: {e}")
    
    def load_model(self):
        """Charge un modèle existant"""
        try:
            filepath = filedialog.askopenfilename(
                title="Charger un modèle",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filepath:
                if self.network is None:
                    self.create_model()
                
                if self.network.load_model(filepath):
                    self.log(f"✅ Modèle chargé: {filepath}")
                else:
                    self.log(f"❌ Échec du chargement: {filepath}")
        except Exception as e:
            self.log(f"❌ Erreur chargement: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
    
    def save_model(self):
        """Sauvegarde le modèle"""
        if self.network is None:
            messagebox.showwarning("Attention", "Aucun modèle à sauvegarder!")
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                title="Sauvegarder le modèle",
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
            if filepath:
                self.network.save_model(filepath)
                self.log(f"✅ Modèle sauvegardé: {filepath}")
        except Exception as e:
            self.log(f"❌ Erreur sauvegarde: {e}")
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {e}")
    
    def load_mnist(self):
        """Charge les données MNIST"""
        def load_data():
            try:
                self.log("📥 Chargement MNIST en cours...")
                import tensorflow as tf
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                
                # Preprocessing
                x_train = x_train.astype('float32') / 255.0
                x_test = x_test.astype('float32') / 255.0
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_test = x_test.reshape(x_test.shape[0], -1)
                
                # One-hot encoding
                y_train_onehot = np.zeros((y_train.shape[0], 10))
                y_test_onehot = np.zeros((y_test.shape[0], 10))
                for i in range(y_train.shape[0]):
                    y_train_onehot[i, y_train[i]] = 1
                for i in range(y_test.shape[0]):
                    y_test_onehot[i, y_test[i]] = 1
                
                self.mnist_data = (x_train, y_train_onehot), (x_test, y_test_onehot)
                self.log(f"✅ MNIST chargé: {x_train.shape[0]} échantillons d'entraînement")
            except Exception as e:
                self.log(f"❌ Erreur chargement MNIST: {e}")
        
        # Lancer dans un thread séparé
        threading.Thread(target=load_data, daemon=True).start()
    
    # def load_image(self):
    #     """Charge une image pour test"""
    #     try:
    #         filepath = filedialog.askopenfilename(
    #             title="Charger une image",
    #             filetypes=[("Images", "*.png *.jpg *.jpeg *.npy"), ("All files", "*.*")]
    #         )
    #         if filepath:
    #             # Adapter selon votre classe ImportImage
    #             import_img = import_image()
    #             if filepath.endswith('.npy'):
    #                 self.current_image = import_img.import_image_npy(filepath)
    #             else:
    #                 self.current_image = import_img.import_image_png(filepath)
                
    #             self.current_image = self.current_image.flatten() / 255.0
    #             self.log(f"✅ Image chargée: {filepath}")
    #     except Exception as e:
    #         self.log(f"❌ Erreur chargement image: {e}")
    #         messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")



    def load_image(self):
        """Charge une image pour test, la centre et la normalise"""
        try:
            filepath = filedialog.askopenfilename(
                title="Charger une image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.npy"), ("All files", "*.*")]
            )
            if filepath:
                import_img = import_image()
                if filepath.endswith('.npy'):
                    img = import_img.import_image_npy(filepath)
                else:
                    img = import_img.import_image_png(filepath)
                
                if img.shape != (28, 28):
                    raise ValueError("L'image doit être 28x28 pixels")

                # # Inverser les couleurs (MNIST : fond noir, chiffre blanc)
                # img = 1.0 - (img / 255.0)

                # Centrer l'image
                img_centered = center_image(img)

                # Aplatir pour le réseau
                self.current_image = img_centered.flatten()

                self.log(f"✅ Image chargée et centrée: {filepath}")
        except Exception as e:
            self.log(f"❌ Erreur chargement image: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {e}")
            print(e)


    def Create_image(self):
        from DrawNumber import main  
        main()

    
    def train_simple(self):
        """Entraînement simple avec image courante"""
        if self.network is None:
            messagebox.showwarning("Attention", "Créez d'abord un modèle!")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image!")
            return
        
        def train():
            try:
                epochs = int(self.epochs_var.get())
                lr = float(self.lr_var.get())
                self.network.learning_rate = lr
                
                self.training_active = True
                self.log(f"🚀 Début entraînement simple: {epochs} époques")
                
                for epoch in range(epochs):
                    if not self.training_active:
                        break
                    
                    result = self.network.Go(self.current_image)
                    
                    if epoch % 10 == 0:
                        self.log(f"Époque {epoch}/{epochs}")
                
                self.log("✅ Entraînement simple terminé!")
            except Exception as e:
                self.log(f"❌ Erreur entraînement: {e}")
        
        threading.Thread(target=train, daemon=True).start()
    
    def train_mnist(self):
        """Entraînement sur MNIST"""
        if self.network is None or self.mnist_data is None:
            messagebox.showwarning("Attention", "Créez un modèle et chargez MNIST!")
            return
        
        def train():
            try:
                epochs = int(self.epochs_var.get())
                lr = float(self.lr_var.get())
                self.network.learning_rate = lr
                
                (x_train, y_train), _ = self.mnist_data
                
                self.training_active = True
                self.log(f"📊 Début entraînement MNIST: {epochs} époques")
                
                for epoch in range(epochs):
                    if not self.training_active:
                        break
                    
                    # Échantillon aléatoire
                    idx = np.random.randint(0, len(x_train))
                    result = self.network.training(x_train[idx], y_train[idx])
                    
                    if epoch % 10 == 0:
                        accuracy = self.calculate_accuracy()
                        self.log(f"Époque {epoch}/{epochs} - Précision: {accuracy:.2f}%")
                
                self.log("✅ Entraînement MNIST terminé!")
            except Exception as e:
                self.log(f"❌ Erreur entraînement MNIST: {e}")
        
        threading.Thread(target=train, daemon=True).start()
    
    def stop_training(self):
        """Arrête l'entraînement"""
        self.training_active = False
        self.log("⏹️ Arrêt de l'entraînement demandé")
    
    def predict_current(self):
        """Prédit l'image courante"""
        if self.network is None or self.current_image is None:
            messagebox.showwarning("Attention", "Chargez un modèle et une image!")
            return
        
        try:
            result = self.network.Go(self.current_image)
            prediction = np.argmax(result)
            confidence = result[prediction] * 100
            
            self.log(f"🎯 Prédiction: {prediction} (confiance: {confidence:.2f}%)")
            
            # Afficher toutes les probabilités
            for i, prob in enumerate(result):
                self.log(f"  {i}: {prob*100:.2f}%")
                
        except Exception as e:
            self.log(f"❌ Erreur prédiction: {e}")
    
    def test_mnist(self):
        """Test sur un échantillon MNIST"""
        if self.network is None or self.mnist_data is None:
            messagebox.showwarning("Attention", "Chargez un modèle et MNIST!")
            return
        
        try:
            _, (x_test, y_test) = self.mnist_data
            
            # Tester sur 100 échantillons
            correct = 0
            total = min(100, len(x_test))
            
            self.log(f"📈 Test sur {total} échantillons...")
            
            for i in range(total):
                result = self.network.Go(x_test[i])
                prediction = np.argmax(result)
                actual = np.argmax(y_test[i])
                
                if prediction == actual:
                    correct += 1
            
            accuracy = (correct / total) * 100
            self.log(f"✅ Précision sur test: {accuracy:.2f}% ({correct}/{total})")
            
        except Exception as e:
            self.log(f"❌ Erreur test MNIST: {e}")
    
    def calculate_accuracy(self):
        """Calcule la précision sur un petit échantillon"""
        try:
            if self.mnist_data is None:
                return 0.0
            
            (x_train, y_train), _ = self.mnist_data
            correct = 0
            total = 10  # Test rapide sur 10 échantillons
            
            for i in range(total):
                idx = np.random.randint(0, len(x_train))
                result = self.network.Go(x_train[idx])
                prediction = np.argmax(result)
                actual = np.argmax(y_train[idx])
                
                if prediction == actual:
                    correct += 1
            
            return (correct / total) * 100
        except:
            return 0.0

def main():
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()






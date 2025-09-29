# Application de dessin pour créer des images 28x28 pixels centrées comme MNIST
# L'application permet de dessiner, centrer automatiquement, et sauvegarder l'image

import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt

class DrawingCanvas:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Toile de dessin 28x28 pour réseau de neurones (avec centrage MNIST)")
        self.root.resizable(False, False)
        
        # Dimensions
        self.canvas_size = 28  # 28x28 = 784 pixels
        self.pixel_size = 15   # Taille d'affichage de chaque pixel
        self.display_size = self.canvas_size * self.pixel_size
        
        # Array pour stocker l'image (0=blanc, 255=noir)
        self.image_array = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        # Variables pour le dessin
        self.drawing = False
        self.brush_size = 1
        
        # Options de centrage
        self.auto_center = tk.BooleanVar(value=True)
        self.mnist_style = tk.BooleanVar(value=False)  # False = style original, True = style MNIST inversé
        
        self.setup_ui()
    
    def center_image_like_mnist(self, image_array, target_size=(28, 28)):
        """
        Centre une image comme MNIST avec option de style
        """
        # Convertir en PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array, mode='L')
        img_array = np.array(image)
        
        # Appliquer le style MNIST si demandé (inverser les couleurs)
        if self.mnist_style.get():
            img_array = 255 - img_array
            background_color = 0  # Fond noir pour MNIST
        else:
            background_color = 0  # Fond blanc pour style original
        
        # Rogner les bords vides
        # Pour style original: chercher pixels noirs (> 0)
        # Pour style MNIST: chercher pixels blancs (> 0 après inversion)
        rows = np.any(img_array > 50, axis=1)  # Seuil pour éviter le bruit
        cols = np.any(img_array > 50, axis=0)
        
        if np.any(rows) and np.any(cols):
            # Trouver les indices de la boîte englobante
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Extraire la région d'intérêt
            roi = img_array[y_min:y_max+1, x_min:x_max+1]
            
            # Calculer les dimensions
            h, w = roi.shape
            
            # Calculer le facteur d'échelle (contenu occupe ~80% de l'image finale)
            scale_factor = min(20/h, 20/w)  # 20 sur 28 ≈ 71%
            
            # Nouvelles dimensions
            new_h = max(1, int(h * scale_factor))
            new_w = max(1, int(w * scale_factor))
            
            # Redimensionner avec PIL
            roi_pil = Image.fromarray(roi)
            roi_resized = roi_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            roi_array = np.array(roi_resized)
            
            # Créer l'image finale avec la couleur de fond appropriée
            result = np.full(target_size, background_color, dtype=np.uint8)
            
            # Calculer la position pour centrer
            start_y = (target_size[0] - new_h) // 2
            start_x = (target_size[1] - new_w) // 2
            
            # Placer l'image redimensionnée au centre
            end_y = start_y + roi_array.shape[0]
            end_x = start_x + roi_array.shape[1]
            
            result[start_y:end_y, start_x:end_x] = roi_array
            
            return result
        else:
            return np.full(target_size, background_color, dtype=np.uint8)
        
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Canvas pour dessiner
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.display_size, 
            height=self.display_size,
            bg='white',
            cursor='crosshair'
        )
        self.canvas.pack()
        
        # Dessiner la grille
        self.draw_grid()
        
        # Events de souris
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Frame pour les contrôles
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(pady=10)
        
        # Boutons
        tk.Button(controls_frame, text="Effacer", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Centrer maintenant", command=self.center_current_image).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Sauvegarder Array", command=self.save_array).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Afficher Array", command=self.show_array).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Sauvegarder Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Frame pour les options
        options_frame = tk.Frame(main_frame)
        options_frame.pack(pady=5)
        
        # Checkbox pour centrage automatique
        tk.Checkbutton(options_frame, text="Centrage automatique lors de la sauvegarde", 
                      variable=self.auto_center).pack(anchor='w')
        
        # Checkbox pour style MNIST
        tk.Checkbutton(options_frame, text="Style MNIST (fond noir, contenu blanc)", 
                      variable=self.mnist_style).pack(anchor='w')
        
        # Frame pour la taille du pinceau
        brush_frame = tk.Frame(main_frame)
        brush_frame.pack(pady=5)
        
        tk.Label(brush_frame, text="Taille pinceau:").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=1)
        brush_scale = tk.Scale(brush_frame, from_=1, to=3, orient=tk.HORIZONTAL, variable=self.brush_var)
        brush_scale.pack(side=tk.LEFT, padx=5)
        brush_scale.bind('<Motion>', self.update_brush_size)
        
        # Label d'information
        info_frame = tk.Frame(main_frame)
        info_frame.pack(pady=5)
        tk.Label(info_frame, text=f"Résolution: {self.canvas_size}x{self.canvas_size} = {self.canvas_size**2} pixels", 
        font=('Arial', 10)).pack()
        tk.Label(info_frame, text="Style original: fond blanc, contenu noir | Style MNIST: fond noir, contenu blanc", 
        font=('Arial', 9), fg='blue').pack()
        
    def draw_grid(self):
        """Dessine une grille légère pour visualiser les pixels"""
        for i in range(self.canvas_size + 1):
            x = i * self.pixel_size
            self.canvas.create_line(x, 0, x, self.display_size, fill='lightgray', width=1)
            self.canvas.create_line(0, x, self.display_size, x, fill='lightgray', width=1)
    
    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def draw(self, event):
        if not self.drawing:
            return
            
        # Convertir les coordonnées de la souris en coordonnées de pixel
        pixel_x = int(event.x // self.pixel_size)
        pixel_y = int(event.y // self.pixel_size)
        
        # Vérifier les limites
        if 0 <= pixel_x < self.canvas_size and 0 <= pixel_y < self.canvas_size:
            brush_size = self.brush_var.get()
            
            # Dessiner avec la taille de pinceau spécifiée
            for dx in range(-brush_size//2, brush_size//2 + 1):
                for dy in range(-brush_size//2, brush_size//2 + 1):
                    px, py = pixel_x + dx, pixel_y + dy
                    if 0 <= px < self.canvas_size and 0 <= py < self.canvas_size:
                        # Mettre à jour l'array (0 = blanc, 255 = noir)
                        self.image_array[py, px] = 255  
                        
                        # Dessiner sur le canvas
                        x1 = px * self.pixel_size
                        y1 = py * self.pixel_size
                        x2 = x1 + self.pixel_size
                        y2 = y1 + self.pixel_size
                        
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')
    
    def update_brush_size(self, event):
        self.brush_size = self.brush_var.get()
    
    def center_current_image(self):
        """Centre l'image actuelle et met à jour l'affichage"""
        if np.sum(self.image_array) == 0:
            messagebox.showwarning("Attention", "Aucun contenu à centrer !")
            return
        
        # Centrer l'image
        centered_array = self.center_image_like_mnist(self.image_array)
        self.image_array = centered_array
        
        # Redessiner le canvas
        self.redraw_canvas()
        
        print("Image centrée selon le style MNIST")
    
    def redraw_canvas(self):
        """Redessine le canvas basé sur l'array actuel"""
        self.canvas.delete("all")
        self.draw_grid()
        
        # Redessiner tous les pixels noirs
        for y in range(self.canvas_size):
            for x in range(self.canvas_size):
                if self.image_array[y, x] > 0:
                    x1 = x * self.pixel_size
                    y1 = y * self.pixel_size
                    x2 = x1 + self.pixel_size
                    y2 = y1 + self.pixel_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='black')
    
    def clear_canvas(self):
        """Efface la toile"""
        self.canvas.delete("all")
        self.draw_grid()
        self.image_array = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        print("Toile effacée")
    
    def save_array(self):
        """Sauvegarde l'array dans un fichier .npy"""
        try:
            # Centrer automatiquement si l'option est activée
            array_to_save = self.image_array
            if self.auto_center.get() and np.sum(self.image_array) > 0:
                array_to_save = self.center_image_like_mnist(self.image_array)
                print("Image centrée automatiquement avant sauvegarde")
            
            # Normaliser l'array (0-1) comme souvent utilisé pour les réseaux de neurones
            normalized_array = array_to_save / 255.0
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".npy",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            
            if filename:
                np.save(filename, normalized_array)
                print(f"Array sauvegardé: {filename}")
                print(f"Shape: {normalized_array.shape}")
                print(f"Min: {normalized_array.min():.3f}, Max: {normalized_array.max():.3f}")
                messagebox.showinfo("Succès", f"Array sauvegardé dans {filename}\n(Centré automatiquement: {self.auto_center.get()})")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def show_array(self):
        """Affiche l'array sous forme de graphique"""
        try:
            # Préparer les données
            original_array = self.image_array / 255.0
            
            # Centrer si demandé
            centered_array = None
            if np.sum(self.image_array) > 0:
                centered_array = self.center_image_like_mnist(self.image_array) / 255.0
            
            # Créer la figure
            if centered_array is not None:
                plt.figure(figsize=(12, 8))
                
                # Image originale
                plt.subplot(2, 3, 1)
                plt.imshow(original_array, cmap='gray', interpolation='nearest')
                plt.title('Image actuelle')
                plt.axis('off')
                
                # Image centrée
                plt.subplot(2, 3, 2)
                plt.imshow(centered_array, cmap='gray', interpolation='nearest')
                plt.title('Version centrée (MNIST)')
                plt.axis('off')
                
                # Comparaison côte à côte
                plt.subplot(2, 3, 3)
                comparison = np.hstack([original_array, np.ones((28, 2)), centered_array])
                plt.imshow(comparison, cmap='gray', interpolation='nearest')
                plt.title('Comparaison (Original | Centré)')
                plt.axis('off')
                
                # Histogrammes
                plt.subplot(2, 3, 4)
                plt.hist(original_array.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
                plt.title('Distribution - Original')
                plt.xlabel('Valeur')
                plt.ylabel('Fréquence')
                
                plt.subplot(2, 3, 5)
                plt.hist(centered_array.flatten(), bins=50, alpha=0.7, color='red', label='Centré')
                plt.title('Distribution - Centré')
                plt.xlabel('Valeur')
                plt.ylabel('Fréquence')
                
                # Statistiques
                plt.subplot(2, 3, 6)
                plt.text(0.1, 0.8, f"=== IMAGE ORIGINALE ===", fontsize=10, fontweight='bold')
                plt.text(0.1, 0.7, f"Pixels non-blancs: {np.sum(original_array > 0)}", fontsize=9)
                plt.text(0.1, 0.6, f"Moyenne: {original_array.mean():.3f}", fontsize=9)
                plt.text(0.1, 0.5, f"Max: {original_array.max():.3f}", fontsize=9)
                
                plt.text(0.1, 0.3, f"=== IMAGE CENTRÉE ===", fontsize=10, fontweight='bold')
                plt.text(0.1, 0.2, f"Pixels non-blancs: {np.sum(centered_array > 0)}", fontsize=9)
                plt.text(0.1, 0.1, f"Moyenne: {centered_array.mean():.3f}", fontsize=9)
                plt.text(0.1, 0.0, f"Max: {centered_array.max():.3f}", fontsize=9)
                plt.axis('off')
                
            else:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(original_array, cmap='gray', interpolation='nearest')
                plt.title(f'Image {self.canvas_size}x{self.canvas_size}')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.hist(original_array.flatten(), bins=50, alpha=0.7, color='blue')
                plt.title('Distribution des pixels')
                plt.xlabel('Valeur (0=blanc, 1=noir)')
                plt.ylabel('Fréquence')
            
            plt.tight_layout()
            plt.show()
            
            # Afficher les statistiques dans la console
            print(f"\n=== Statistiques de l'image ===")
            print(f"Image originale - Pixels non-blancs: {np.sum(original_array > 0)}")
            if centered_array is not None:
                print(f"Image centrée - Pixels non-blancs: {np.sum(centered_array > 0)}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage: {str(e)}")
    
    def save_image(self):
        """Sauvegarde l'image en format PNG"""
        try:
            # Centrer automatiquement si l'option est activée
            array_to_save = self.image_array
            if self.auto_center.get() and np.sum(self.image_array) > 0:
                array_to_save = self.center_image_like_mnist(self.image_array)
                print("Image centrée automatiquement avant sauvegarde")
            
            # Créer une image PIL directement depuis l'array
            img = Image.fromarray(array_to_save, mode='L')
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if filename:
                img.save(filename)
                messagebox.showinfo("Succès", f"Image sauvegardée dans {filename}\n(Centrée automatiquement: {self.auto_center.get()})")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def get_array(self, centered=None):
        """Retourne l'array normalisé (0-1) pour utilisation directe"""
        if centered is None:
            centered = self.auto_center.get()
        
        if centered and np.sum(self.image_array) > 0:
            return self.center_image_like_mnist(self.image_array) / 255.0
        else:
            return self.image_array / 255.0
    
    def run(self):
        """Lance l'application"""
        print("=== Toile de dessin pour réseau de neurones avec centrage MNIST ===")
        print(f"Résolution: {self.canvas_size}x{self.canvas_size} = {self.canvas_size**2} pixels")
        print("Instructions:")
        print("- Cliquez et glissez pour dessiner")
        print("- Utilisez 'Centrer maintenant' pour centrer l'image actuelle")
        print("- Le centrage automatique est activé par défaut lors des sauvegardes")
        print("- L'image sera centrée comme dans le dataset MNIST")
        print("- L'array est normalisé entre 0 (blanc) et 1 (noir)")
        
        self.root.mainloop()

# Fonction pour utiliser le programme
def main():
    app = DrawingCanvas()
    app.run()

# Pour utilisation directe dans un notebook ou script
def create_drawing_app():
    """Créer et retourner une instance de l'application"""
    return DrawingCanvas()

if __name__ == "__main__":
    main()
#La totalite de ce code a ete generee par l'IA, il s'agit d'une application de dessin pour créer des images 28x28 pixels, 
# typiquement utilisées pour les réseaux de neurones comme MNIST. 
# L'application permet de dessiner, sauvegarder l'image en format .npy ou .png, pour tester mon modèles de machine learning.


import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class DrawingCanvas:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Toile de dessin 28x28 pour réseau de neurones")
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
        
        self.setup_ui()
        
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
        tk.Button(controls_frame, text="Sauvegarder Array", command=self.save_array).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Afficher Array", command=self.show_array).pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Sauvegarder Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
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
    
    def clear_canvas(self):
        """Efface la toile"""
        self.canvas.delete("all")
        self.draw_grid()
        self.image_array = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        print("Toile effacée")
    
    def save_array(self):
        """Sauvegarde l'array dans un fichier .npy"""
        try:
            # Normaliser l'array (0-1) comme souvent utilisé pour les réseaux de neurones
            normalized_array = self.image_array / 255.0  # Normaliser directement
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".npy",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
            )
            
            if filename:
                np.save(filename, normalized_array)
                print(f"Array sauvegardé: {filename}")
                print(f"Shape: {normalized_array.shape}")
                print(f"Min: {normalized_array.min():.3f}, Max: {normalized_array.max():.3f}")
                messagebox.showinfo("Succès", f"Array sauvegardé dans {filename}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def show_array(self):
        """Affiche l'array sous forme de graphique"""
        try:
            # Créer une version normalisée pour l'affichage
            display_array = self.image_array / 255.0
            
            plt.figure(figsize=(8, 6))
            
            # Affichage de l'image
            plt.subplot(1, 2, 1)
            plt.imshow(display_array, cmap='gray', interpolation='nearest')
            plt.title(f'Image {self.canvas_size}x{self.canvas_size}')
            plt.axis('off')
            
            # Histogramme des valeurs
            plt.subplot(1, 2, 2)
            plt.hist(display_array.flatten(), bins=50, alpha=0.7, color='blue')
            plt.title('Distribution des pixels')
            plt.xlabel('Valeur (0=blanc, 1=noir)')
            plt.ylabel('Fréquence')
            
            plt.tight_layout()
            plt.show()
            
            # Afficher aussi les statistiques dans la console
            print(f"\n=== Statistiques de l'image ===")
            print(f"Shape: {display_array.shape}")
            print(f"Min: {display_array.min():.3f}")
            print(f"Max: {display_array.max():.3f}")
            print(f"Moyenne: {display_array.mean():.3f}")
            print(f"Pixels non-blancs: {np.sum(display_array > 0)}")
            print(f"Array flatten shape: {display_array.flatten().shape}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage: {str(e)}")
    
    def save_image(self):
        """Sauvegarde l'image en format PNG"""
        try:
            # Créer une image PIL directement depuis l'array
            img = Image.fromarray(self.image_array, mode='L')
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if filename:
                img.save(filename)
                messagebox.showinfo("Succès", f"Image sauvegardée dans {filename}")
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")
    
    def get_array(self):
        """Retourne l'array normalisé (0-1) pour utilisation directe"""
        return self.image_array / 255.0
    
    def run(self):
        """Lance l'application"""
        print("=== Toile de dessin pour réseau de neurones ===")
        print(f"Résolution: {self.canvas_size}x{self.canvas_size} = {self.canvas_size**2} pixels")
        print("Instructions:")
        print("- Cliquez et glissez pour dessiner")
        print("- Ajustez la taille du pinceau avec le slider")
        print("- Utilisez 'Sauvegarder Array' pour exporter en .npy")
        print("- Utilisez 'Afficher Array' pour voir les statistiques")
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

# Exemple d'utilisation programmatique:
"""
# Créer l'application
app = create_drawing_app()

# Lancer l'interface
app.run()

# Après avoir dessiné, récupérer l'array:
image_array = app.get_array()  # Array 28x28 normalisé entre 0 et 1
print(f"Shape pour réseau de neurones: {image_array.shape}")
print(f"Array flatten: {image_array.flatten().shape}")  # (784,) pour input layer
"""
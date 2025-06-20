from PIL import Image
import numpy as np

class import_image:
    def __init__(self, size=(28,28)):
        self.size = size

    def import_image_png(self, image_path):

        
        # Ouvrir l'image
        image = Image.open(image_path)
        
        # Redimensionner l'image
        image = image.resize(self.size)
        
        # Convertir l'image en tableau numpy
        image_array = np.array(image)
        
        # Normaliser les valeurs des pixels entre 0 et 1
        image_array = image_array / 255.0
        
        return image_array
    

    def import_image_npy(self, image_path):
        
        return np.load(image_path)





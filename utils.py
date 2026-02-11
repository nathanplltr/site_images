import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def get_colors_palette(image_path, k=3):
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((150, 150))
    pixels = np.array(img).reshape(-1, 3)
    
    # On utilise le 'k' choisi par l'utilisateur
    model = KMeans(n_clusters=int(k), n_init=10)
    model.fit(pixels)
    
    colors = model.cluster_centers_.astype(int)
    # On transforme chaque couleur en format HEX
    return ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in colors]
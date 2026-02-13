import os
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from PIL import Image
from sklearn.cluster import KMeans
import uuid
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)

# Dossiers de travail
IMAGE_FOLDER = 'uploads'
STATIC_RESULTS = 'static/results'

# Création des dossiers s'ils n'existent pas
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(STATIC_RESULTS, exist_ok=True)

# --- Fonctions Utilitaires ---

def get_colors_palette(image_path, k=3):
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((150, 150))
    pixels = np.array(img).reshape(-1, 3)
    model = KMeans(n_clusters=int(k), n_init=10)
    model.fit(pixels)
    colors = model.cluster_centers_.astype(int)
    return ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in colors]

# --- Routes Galerie ---

@app.route('/')
def index():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().split('.')[-1] in ['jpg', 'jpeg', 'png', 'gif', 'webp']]
    return render_template('index.html', images=images)

@app.route('/get_analysis')
def get_analysis():
    filename = request.args.get('filename')
    k_value = request.args.get('k', 3)
    path = os.path.join(IMAGE_FOLDER, filename)
    palette = get_colors_palette(path, k=k_value)
    return jsonify({'palette': palette})

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# --- Routes Segmentation (Scan de dossier) ---

@app.route('/segmentation')
def segmentation_page():
    return render_template('segmentation.html')

@app.route('/scan_folder', methods=['POST'])
def scan_folder():
    path = request.json.get('path')
    if not os.path.exists(path):
        return jsonify({'error': 'Chemin introuvable ou inaccessible'}), 404
    
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify({'files': files})

@app.route('/run_segmentation_path', methods=['POST'])
def run_segmentation_path():
    data = request.json
    folder_path = data.get('path')
    filename = data.get('filename')
    k = int(data.get('k', 3))
    
    full_path = os.path.join(folder_path, filename)
    
    # 1. Traitement de l'image
    img = Image.open(full_path).convert('RGB')
    img.thumbnail((400, 400)) # Optimisation pour ton Mac
    
    img_np = np.array(img)
    w, h, d = img_np.shape
    pixels = img_np.reshape(-1, 3)

    # 2. Algorithme K-Means
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # 3. Création de l'image segmentée
    new_pixels = kmeans.cluster_centers_.astype(np.uint8)[labels]
    segmented_img = new_pixels.reshape(w, h, d)

    # --- MODIFICATION ICI : NOM FIXE POUR ÉVITER L'ACCUMULATION ---
    result_filename = "latest_segmentation.png"
    result_path = os.path.join(STATIC_RESULTS, result_filename)
    
    # On écrase l'ancien fichier s'il existe
    Image.fromarray(segmented_img).save(result_path)

    # On ajoute un timestamp pour tromper le cache du navigateur
    import time
    timestamp = int(time.time())
    
    return jsonify({'result_url': f'/static/results/{result_filename}?v={timestamp}'})

# --- Route pour l'Approche Hiérarchique ---

@app.route('/hierarchie')
def hierarchie_page():
    return render_template('hierarchie.html')

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

@app.route('/run_hierarchical', methods=['POST'])
def run_hierarchical():
    import time # Assure-toi qu'il est bien importé
    data = request.json
    folder_path = data.get('path')
    filename = data.get('filename')
    threshold = int(data.get('threshold', 5)) 
    
    full_path = os.path.join(folder_path, filename)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'Fichier introuvable'}), 404

    img = Image.open(full_path).convert('RGB')
    img.thumbnail((80, 80)) 
    img_np = np.array(img)
    w, h, d = img_np.shape
    pixels = img_np.reshape(-1, 3)
    
    # 1. Calcul du linkage
    Z = linkage(pixels, method='ward')

    # 2. Dendrogramme
    plt.figure(figsize=(20, 15))
    dendrogram(Z, truncate_mode='lastp', p=50, leaf_rotation=90., leaf_font_size=8.)
    plt.title("Structure hiérarchique")
    tree_path = os.path.join(STATIC_RESULTS, "latest_hierarchy_tree.png")
    plt.savefig(tree_path)
    plt.close()

    # 3. Segmentation (On utilise le nombre réel de clusters trouvés)
    labels = fcluster(Z, t=threshold, criterion='maxclust')
    unique_labels = np.unique(labels)
    
    new_pixels = np.zeros_like(pixels)
    for lbl in unique_labels:
        indices = np.where(labels == lbl)
        mean_color = pixels[indices].mean(axis=0)
        new_pixels[indices] = mean_color

    segmented_img = new_pixels.reshape(w, h, d).astype(np.uint8)
    seg_path = os.path.join(STATIC_RESULTS, "latest_hierarchy_seg.png")
    Image.fromarray(segmented_img).save(seg_path)

    ts = int(time.time())
    return jsonify({
        'tree_url': f'/static/results/latest_hierarchy_tree.png?v={ts}',
        'seg_url': f'/static/results/latest_hierarchy_seg.png?v={ts}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=True)


from sklearn.cluster import DBSCAN

# --- Routes DBSCAN ---

@app.route('/dbscan')
def dbscan_page():
    return render_template('dbscan.html')

@app.route('/run_dbscan', methods=['POST'])
def run_dbscan():
    data = request.json
    folder_path = data.get('path')
    filename = data.get('filename')
    # eps: distance max entre deux points pour être voisins
    # min_samples: nombre de points pour former un groupe
    eps = float(data.get('eps', 5.0))
    min_samples = int(data.get('min_samples', 10))
    
    full_path = os.path.join(folder_path, filename)
    img = Image.open(full_path).convert('RGB')
    img.thumbnail((100, 100)) # On reste sur une petite taille pour la fluidité
    img_np = np.array(img)
    w, h, d = img_np.shape
    pixels = img_np.reshape(-1, 3)

    # Lancement de DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
    labels = db.labels_

    # Nombre de clusters (en excluant le bruit codé -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Coloration des clusters avec la moyenne
    new_pixels = np.zeros_like(pixels)
    for lbl in set(labels):
        indices = np.where(labels == lbl)
        if lbl == -1:
            new_pixels[indices] = [0, 0, 0] # Le bruit en noir
        else:
            new_pixels[indices] = pixels[indices].mean(axis=0)

    segmented_img = new_pixels.reshape(w, h, d).astype(np.uint8)
    result_filename = "latest_dbscan.png"
    result_path = os.path.join(STATIC_RESULTS, result_filename)
    Image.fromarray(segmented_img).save(result_path)

    import time
    return jsonify({
        'result_url': f'/static/results/{result_filename}?v={int(time.time())}',
        'n_clusters': n_clusters
    })
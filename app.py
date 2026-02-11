from flask import Flask, render_template, send_from_directory, jsonify, request
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import uuid 

app = Flask(__name__)
IMAGE_FOLDER = 'uploads'
STATIC_FOLDER = 'static/results'

# S'assure que les dossiers existent
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# --- Fonction utilitaire pour la palette (si tu ne l'as pas dans utils.py) ---
def get_colors_palette(image_path, k=3):
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((150, 150)) # On réduit pour la rapidité
    pixels = np.array(img).reshape(-1, 3)
    model = KMeans(n_clusters=int(k), n_init=10)
    model.fit(pixels)
    colors = model.cluster_centers_.astype(int)
    return ['#{:02x}{:02x}{:02x}'.format(c[0], c[1], c[2]) for c in colors]

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

@app.route('/segmentation')
def segmentation_page():
    return render_template('segmentation.html')

@app.route('/run_segmentation', methods=['POST'])
def run_segmentation():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image'}), 400
        
    file = request.files['image']
    k = int(request.form.get('k', 3))
    
    if file:
        img = Image.open(file).convert('RGB')
        img_np = np.array(img)
        w, h, d = img_np.shape
        pixels = img_np.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # CORRECTION ICI : np.uint8 au lieu de uint8
        new_pixels = kmeans.cluster_centers_.astype(np.uint8)[labels]
        segmented_img = new_pixels.reshape(w, h, d)

        result_filename = f"seg_{uuid.uuid4()}.png"
        result_path = os.path.join(STATIC_FOLDER, result_filename)
        Image.fromarray(segmented_img).save(result_path)

        return jsonify({'result_url': f'/static/results/{result_filename}'})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
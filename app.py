import os
from flask import Flask, render_template, send_from_directory, jsonify, request
from utils import get_colors_palette # Assure-toi d'avoir créé utils.py avec la fonction

app = Flask(__name__)
IMAGE_FOLDER = 'uploads'

@app.route('/')
def index():
    # On scanne le dossier pour trouver TOUTES les images à chaque chargement
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().split('.')[-1] in ['jpg', 'jpeg', 'png', 'gif', 'webp']]
    
    # On envoie cette liste au fichier HTML
    return render_template('index.html', images=images)

@app.route('/get_analysis')
def get_analysis():
    filename = request.args.get('filename')
    k_value = request.args.get('k', 3)
    
    path = os.path.join(IMAGE_FOLDER, filename)
    # Appel à ton algo K-Means
    palette = get_colors_palette(path, k=k_value)
    
    return jsonify({'palette': palette})

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    # On force le port 5002 si tu as eu des soucis de connexion sur le 5000
    app.run(debug=True, port=5002)
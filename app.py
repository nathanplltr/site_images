import os
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)
IMAGE_FOLDER = 'uploads'

@app.route('/')
def index():
    # On récupère la liste de toutes les images pour le menu déroulant
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return render_template('index.html', images=images)

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
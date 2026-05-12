from flask import Flask, render_template, request
import os
from PIL import Image
import imagehash

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():

    image1 = request.files['image1']
    image2 = request.files['image2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], image1.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], image2.filename)

    image1.save(path1)
    image2.save(path2)

    hash1 = imagehash.average_hash(Image.open(path1))
    hash2 = imagehash.average_hash(Image.open(path2))

    difference = hash1 - hash2

    similarity = max(0, 100 - (difference * 5))

    return render_template(
        'index.html',
        score=similarity,
        image1='/' + path1,
        image2='/' + path2
    )

if __name__ == '__main__':
    app.run(debug=True)
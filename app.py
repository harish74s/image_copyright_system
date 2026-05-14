from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import uuid
from skimage.metrics import structural_similarity as ssim

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

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)

    similarity = round(score * 100, 2)

    diff = (diff * 255).astype(np.uint8)

    thresh = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                img2,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
    heatmap_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        heatmap_filename
    )

    cv2.imwrite(heatmap_path, img2)

    return render_template(
        'index.html',
        score=similarity,
        image1='/' + path1,
        image2='/' + path2,
        heatmap='/' + heatmap_path
    )

if __name__ == '__main__':
    app.run(debug=True)
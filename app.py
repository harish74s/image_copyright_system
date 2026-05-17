from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import uuid
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load ResNet50 pretrained on ImageNet and create a feature-extraction model
base_model = ResNet50(weights='imagenet')
model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

# Build a model that outputs the last convolutional feature maps for spatial comparisons
try:
    conv_layer = base_model.get_layer('conv5_block3_out')
except Exception:
    # fallback: find the last layer with 4D output
    conv_layer = None
    for lyr in reversed(base_model.layers):
        out_shape = getattr(lyr.output, 'shape', None)
        if out_shape is not None and len(out_shape) == 4:
            conv_layer = lyr
            break

conv_model = Model(inputs=base_model.input, outputs=conv_layer.output)
# model that outputs conv maps and final predictions for Grad-CAM
pred_model = Model(inputs=base_model.input, outputs=[conv_layer.output, base_model.output])


def extract_features(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    features = model.predict(img_array)

    return features.flatten()


def compute_spatial_similarity_map(path1, path2, target_size=(224, 224), out_size=(300, 300)):

    def load_preprocess(p):
        img = image.load_img(p, target_size=target_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        return arr

    a1 = load_preprocess(path1)
    a2 = load_preprocess(path2)

    fmap1 = conv_model.predict(a1)[0]  # H, W, C
    fmap2 = conv_model.predict(a2)[0]

    h, w, c = fmap1.shape

    f1 = fmap1.reshape((-1, c))
    f2 = fmap2.reshape((-1, c))

    # normalize per-location feature vectors
    f1_norm = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-10)
    f2_norm = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-10)

    # cosine similarity per spatial location (same spatial coordinate)
    sim = np.sum(f1_norm * f2_norm, axis=1)
    sim_map = sim.reshape((h, w))

    # resize similarity map to output visualization size
    sim_map_resized = cv2.resize(sim_map, out_size)

    # normalize to 0-255
    sim_min, sim_max = sim_map_resized.min(), sim_map_resized.max()
    sim_norm = ((sim_map_resized - sim_min) / (sim_max - sim_min + 1e-10) * 255).astype(np.uint8)

    # difference map for thresholding: low similarity => high diff
    diff_map = 255 - sim_norm

    return sim_norm, diff_map


def compute_gradcam(path, target_size=(224, 224), out_size=(300, 300)):
    img = image.load_img(path, target_size=target_size)
    arr = image.img_to_array(img)
    arr_exp = np.expand_dims(arr, axis=0)
    arr_pre = preprocess_input(arr_exp)

    # get conv maps and predictions
    conv_outputs, preds = pred_model.predict(arr_pre)
    preds = preds[0]
    class_idx = int(np.argmax(preds))

    # compute gradient of the top predicted class score w.r.t. conv outputs
    arr_tensor = tf.convert_to_tensor(arr_pre)
    with tf.GradientTape() as tape:
        tape.watch(arr_tensor)
        conv_outs, predictions = pred_model(arr_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outs)

    # compute channel-wise mean of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    conv_outs = conv_outs[0].numpy()
    # weight conv channels by pooled grads
    for i in range(pooled_grads.shape[-1]):
        conv_outs[:, :, i] *= pooled_grads[i]

    cam = np.sum(conv_outs, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-10)
    cam_resized = cv2.resize(cam, out_size)
    cam_uint8 = (cam_resized * 255).astype(np.uint8)

    return cam_uint8

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

    # create smaller copies for visualization / diff detection
    img1_small = cv2.resize(img1, (300, 300))
    img2_small = cv2.resize(img2, (300, 300))

    gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)

    # Extract deep features (embeddings) and compute cosine similarity
    features1 = extract_features(path1)
    features2 = extract_features(path2)

    similarity = cosine_similarity(
        [features1],
        [features2]
    )[0][0]

    similarity = round(similarity * 100, 2)

    # For learned localization, compute a spatial similarity map from conv feature maps
    sim_norm, diff_map = compute_spatial_similarity_map(path1, path2, out_size=(300, 300))

    # create a heatmap overlay from similarity and Grad-CAM, then combine
    gradcam_map = compute_gradcam(path2, out_size=(300, 300))

    sim_color = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
    grad_color = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)

    combined_color = cv2.addWeighted(sim_color, 0.5, grad_color, 0.5, 0)

    overlay = cv2.addWeighted(img2_small, 0.6, combined_color, 0.4, 0)

    # smooth noisy activations before contour detection
    diff_map = cv2.GaussianBlur(diff_map, (11, 11), 0)

    # threshold diff_map to find differing regions
    thresh = cv2.threshold(
        diff_map,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            overlay,
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

    # save overlay heatmap
    cv2.imwrite(heatmap_path, overlay)

    return render_template(
        'index.html',
        score=similarity,
        image1='/' + path1,
        image2='/' + path2,
        heatmap='/' + heatmap_path
    )

if __name__ == '__main__':
    app.run(debug=True)
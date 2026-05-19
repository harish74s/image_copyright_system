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


def _load_image_pair(path1, path2, target_size=(300, 300)):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise ValueError('Unable to read one or both uploaded images.')

    img1_small = cv2.resize(img1, target_size)
    img2_small = cv2.resize(img2, target_size)

    return img1_small, img2_small


def _normalized_embedding_similarity(features1, features2):
    raw_similarity = float(cosine_similarity([features1], [features2])[0][0])
    return float(np.clip(raw_similarity, 0.0, 1.0))


def _ssim_similarity(img1_small, img2_small):
    gray1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
    value = float(ssim(gray1, gray2))
    return float(np.clip(value, 0.0, 1.0))


def _histogram_similarity(img1_small, img2_small):
    hsv1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [32, 32], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    correlation = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    return float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))


def _estimate_image_category(img_small):
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    mean_bgr = np.mean(img_small.reshape(-1, 3), axis=0)
    color_spread = float(np.std(mean_bgr))
    brightness = float(np.mean(gray))
    saturation = float(np.mean(cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)[:, :, 1]))
    aspect_ratio = float(img_small.shape[1]) / float(img_small.shape[0])

    if edge_density > 0.14 and saturation < 50 and brightness > 110:
        return 'document'

    if edge_density > 0.12 and saturation >= 50:
        return 'screenshot'

    if aspect_ratio > 1.5 and saturation > 45 and edge_density < 0.10:
        return 'landscape'

    if saturation > 55 and color_spread > 25 and edge_density < 0.12:
        return 'photo'

    return 'graphic'


def _apply_category_filter(score, category1, category2):
    if category1 == category2:
        return score, False

    return score * 0.45, True


def _build_verdict(final_score):
    if final_score <= 25:
        return 'Completely Different'
    if final_score <= 50:
        return 'Slight Visual Similarity'
    if final_score <= 75:
        return 'Possibly Modified Copy'
    return 'Highly Similar / Copied'


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

    try:
        img1_small, img2_small = _load_image_pair(path1, path2, target_size=(300, 300))
    except ValueError as exc:
        return render_template('index.html', error=str(exc))

    category1 = _estimate_image_category(img1_small)
    category2 = _estimate_image_category(img2_small)

    features1 = extract_features(path1)
    features2 = extract_features(path2)

    embedding_score = _normalized_embedding_similarity(features1, features2)
    if embedding_score < 0.70:
        embedding_score = 0.0

    ssim_score = _ssim_similarity(img1_small, img2_small)
    histogram_score = _histogram_similarity(img1_small, img2_small)

    final_score = (
        (ssim_score * 0.3) +
        (histogram_score * 0.2) +
        (embedding_score * 0.5)
    )

    final_score, category_mismatch = _apply_category_filter(
        final_score,
        category1,
        category2
    )

    similarity = round(final_score * 100, 2)
    verdict = _build_verdict(similarity)
    if similarity < 60:
        verdict = 'Different Images'

    # For learned localization, compute a spatial similarity map from conv feature maps
    sim_norm, diff_map = compute_spatial_similarity_map(path1, path2, out_size=(300, 300))

    # create a heatmap overlay from similarity and Grad-CAM, then combine
    gradcam_map = compute_gradcam(path2, out_size=(300, 300))

    sim_color = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
    grad_color = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)

    combined_color = cv2.addWeighted(sim_color, 0.5, grad_color, 0.5, 0)

    overlay = cv2.addWeighted(img2_small, 0.75, combined_color, 0.25, 0)

    # smooth noisy activations before contour detection
    diff_map = cv2.GaussianBlur(diff_map, (5, 5), 0)

    # threshold diff_map to find differing regions
    _, thresh = cv2.threshold(
        diff_map,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 2200:
            continue

        if area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue

        contour_mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        contour_area = float(np.count_nonzero(contour_mask))
        if contour_area <= 0:
            continue

        fill_ratio = float(area) / contour_area
        if fill_ratio < 0.35:
            continue

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
        verdict=verdict,
        category1=category1,
        category2=category2,
        category_mismatch=category_mismatch,
        ssim_score=round(ssim_score * 100, 2),
        histogram_score=round(histogram_score * 100, 2),
        embedding_score=round(embedding_score * 100, 2),
        image1='/' + path1,
        image2='/' + path2,
        heatmap='/' + heatmap_path
    )

if __name__ == '__main__':
    app.run(debug=True)
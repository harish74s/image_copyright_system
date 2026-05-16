import os
import numpy as np
import cv2
from app import extract_features

os.makedirs('static/images', exist_ok=True)

# create a simple test image
img = np.zeros((500, 500, 3), dtype=np.uint8)
cv2.putText(img, 'TEST', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 10)
path = 'static/images/test_img.jpg'
cv2.imwrite(path, img)

print('Saved test image to', path)

features = extract_features(path)
print('Feature vector length:', len(features))
print('First 10 values:', features[:10])

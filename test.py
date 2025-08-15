import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ======= CONFIG =======
IMG_PATH = r'E:\lung_tumor\venv\preprocessed_data\test\image\lung_009_slice226.jpg'
MODEL_PATH = r'E:\lung_tumor\venv\unet_lung_model.h5'
IMG_SIZE = 256
THRESHOLD = 0.01  # 1% of image area to consider tumor present

# ======= Load Model =======
model = load_model(MODEL_PATH)

# ======= Load and Preprocess Image =======
image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found at the specified path!")

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image_input = image.astype('float32') / 255.0
image_input = np.expand_dims(image_input, axis=-1)  # Add channel
image_input = np.expand_dims(image_input, axis=0)   # Add batch

# ======= Predict Mask =======
pred_mask = model.predict(image_input)[0, :, :, 0]
binary_mask = (pred_mask > 0.5).astype(np.uint8)

# ======= Tumor Detection Decision =======
tumor_pixels = np.sum(binary_mask)
total_pixels = IMG_SIZE * IMG_SIZE



# ======= Visualize =======
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_mask, cmap='hot')
plt.title('Predicted Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

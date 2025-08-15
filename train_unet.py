import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from build_unet_model import build_unet

# ======== CONFIG ========
IMG_SIZE = 256
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
DATA_DIR = r'C:\Users\Akalya\OneDrive\Desktop\lung_tumor\venv\preprocessed_data'  # Folder from preprocessing step
MODEL_PATH = 'unet_lung_model.h5'

# ======== LOAD IMAGES AND MASKS ========
def load_images_and_masks(split):
    images = []
    masks = []

    img_dir = os.path.join(DATA_DIR, split, 'image')
    mask_dir = os.path.join(DATA_DIR, split, 'mask')

    for filename in tqdm(os.listdir(img_dir), desc=f"Loading {split} data"):
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is not None and mask is not None:
            image = image.astype('float32') / 255.0
            mask = (mask > 127).astype('float32')  # Convert to binary mask

            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)

# ======== LOAD DATA ========
X_train, y_train = load_images_and_masks('train')
X_val, y_val = load_images_and_masks('valid')

print("✅ Dataset loaded")
print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Valid shape: {X_val.shape}, {y_val.shape}")

# ======== SHUFFLE TRAIN DATA ========
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# ======== BUILD & TRAIN MODEL ========
model = build_unet(INPUT_SHAPE)
model.summary()

checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=10,
    epochs=5,
    callbacks=[checkpoint]
)

print("\n✅ Model training complete. Model saved as:", MODEL_PATH)

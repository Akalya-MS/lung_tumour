import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# ========== CONFIG ==========
IMG_SIZE = 256
ROOT_DIR = r'C:\Users\Akalya\OneDrive\Desktop\lung_tumor\new_data'  # Set this to your actual root path
OUTPUT_DIR = 'preprocessed_data'  # Output folder

# ========== AUGMENTATION HELPERS ==========

def random_rotate(img, mask):
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)
    mask_rotated = cv2.warpAffine(mask, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST)
    return img_rotated, mask_rotated

def random_flip(img, mask):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    return img, mask

# ========== PREPROCESS FUNCTION ==========

def preprocess_pair(image_path, mask_path):
    # Load image and mask (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return None, None

    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Removed: Gaussian Blur

    # Random Rotation
    image, mask = random_rotate(image, mask)

    # Random Flip
    image, mask = random_flip(image, mask)

    # Removed: CLAHE

    # Normalize (image to 0-1)
    image = image.astype('float32') / 255.0
    mask = mask.astype('uint8')  # Keep mask as label (0,1)

    return image, mask

# ========== MAIN PROCESS FUNCTION ==========

def process_split(split):
    print(f"Processing {split} data...")
    img_dir = os.path.join(ROOT_DIR, split, 'image')
    mask_dir = os.path.join(ROOT_DIR, split, 'mask')

    out_img_dir = os.path.join(OUTPUT_DIR, split, 'image')
    out_mask_dir = os.path.join(OUTPUT_DIR, split, 'mask')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    for filename in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        image, mask = preprocess_pair(img_path, mask_path)
        if image is not None and mask is not None:
            # Save processed image and mask
            img_out_path = os.path.join(out_img_dir, filename)
            mask_out_path = os.path.join(out_mask_dir, filename)

            # Save image (scale back to 0-255)
            image_uint8 = (image * 255).astype(np.uint8)
            cv2.imwrite(img_out_path, image_uint8)
            cv2.imwrite(mask_out_path, mask)

# ========== RUN ALL SPLITS ==========

if __name__ == "__main__":
    for split in ['train', 'valid', 'test']:
        process_split(split)

    print("\n✅ Preprocessing complete! Files saved in 'preprocessed_data/' folder.")

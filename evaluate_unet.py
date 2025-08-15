import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model

# ========== CONFIG ==========
IMG_SIZE = 256
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
DATA_DIR = 'preprocessed_data'
MODEL_PATH = 'unet_lung_model.h5'

# ========== METRIC FUNCTIONS ==========

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ========== DATA LOADER ==========

def load_test_data():
    images = []
    masks = []
    test_img_dir = os.path.join(DATA_DIR, 'test', 'image')
    test_mask_dir = os.path.join(DATA_DIR, 'test', 'mask')

    for filename in tqdm(os.listdir(test_img_dir), desc="Loading test data"):
        img_path = os.path.join(test_img_dir, filename)
        mask_path = os.path.join(test_mask_dir, filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is not None and mask is not None:
            image = image.astype('float32') / 255.0
            mask = (mask > 127).astype('float32')

            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)

# ========== PLOT ==========

def show_predictions(X, Y_true, Y_pred, num=5):
    for i in range(num):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(Y_true[i].squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(Y_pred[i].squeeze(), cmap='gray')
        plt.axis('off')

        plt.show()

# ========== MAIN ==========
if __name__ == "__main__":
    print("✅ Loading test data...")
    X_test, Y_test = load_test_data()

    print("✅ Loading trained model...")
    model = load_model(MODEL_PATH)

    print("🔍 Predicting on test data...")
    preds = model.predict(X_test)
    preds = (preds > 0.5).astype('float32')  # Threshold to binary mask

    print("📊 Calculating metrics...")
    dice_scores = []
    iou_scores = []

    for i in range(len(X_test)):
        dice = dice_coefficient(Y_test[i], preds[i])
        iou = iou_score(Y_test[i], preds[i])
        dice_scores.append(dice)
        iou_scores.append(iou)

    print(f"\n✅ Average Dice Coefficient: {np.mean(dice_scores):.4f}")
    print(f"✅ Average IoU Score:         {np.mean(iou_scores):.4f}")

    print("\n🖼️ Showing sample predictions...")
    show_predictions(X_test, Y_test, preds, num=5)

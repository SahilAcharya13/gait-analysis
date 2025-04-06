import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from scipy.spatial.distance import cosine

def load_images_from_folder(folder, target_size=(128, 128)):
    """Loads and preprocesses images from a folder."""
    images, filenames = [], sorted(os.listdir(folder))  # Sort filenames to maintain order
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size) / 255.0  # Normalize (0-1 range)
            images.append(img)
    return np.array(images).reshape(-1, 128, 128, 1), filenames

def create_cnn():
    """Defines an optimized CNN model for feature extraction."""
    input_layer = Input(shape=(128, 128, 1))
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(input_layer, x)

def compute_similarity(features1, features2):
    """Computes cosine similarity between two sets of feature vectors."""
    return [1 - cosine(features1[i], features2[i]) for i in range(len(features1))]

def plot_images(images1, images2, filenames1, filenames2, n=5):
    """Plots a sample of images from two angles with improved layout."""
    fig, axes = plt.subplots(2, n, figsize=(12, 5))
    for i in range(n):
        axes[0, i].imshow(images1[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"Angle 1 - {filenames1[i]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(images2[i].squeeze(), cmap="gray")
        axes[1, i].set_title(f"Angle 2 - {filenames2[i]}")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

def plot_similarity(similarities):
    """Plots the cosine similarity scores with enhanced visualization."""
    plt.figure(figsize=(10, 6))
    plt.plot(similarities, marker='o', linestyle='-', color="blue", label="Cosine Similarity")
    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("Similarity Score", fontsize=12)
    plt.title("Cosine Similarity Between Different Angles", fontsize=14)
    plt.axhline(y=np.mean(similarities), color='r', linestyle='--', label="Mean Similarity")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_feature_differences(features1, features2, sample_size=10):
    """Visualizes feature differences using an improved heatmap."""
    feature_diff = np.abs(features1 - features2)
    plt.figure(figsize=(12, 6))
    sns.heatmap(feature_diff[:sample_size], cmap="coolwarm", annot=False, linewidths=0.5)
    plt.title("Feature Differences Between Angles", fontsize=14)
    plt.xlabel("Feature Index", fontsize=12)
    plt.ylabel("Image Index", fontsize=12)
    plt.show()

# Load datasets
folder1 = "D:/Masters/Sem 2/AISNS/Gait_analysis/GaitDatasetB-silh/001/cl-01/018"
folder2 = "D:/Masters/Sem 2/AISNS/Gait_analysis/GaitDatasetB-silh/001/cl-01/036"
images1, filenames1 = load_images_from_folder(folder1)
images2, filenames2 = load_images_from_folder(folder2)

# Ensure equal number of images
min_len = min(len(images1), len(images2))
images1, images2 = images1[:min_len], images2[:min_len]
filenames1, filenames2 = filenames1[:min_len], filenames2[:min_len]

# Create and use CNN feature extractor
cnn_model = create_cnn()
cnn_model.summary()
features1 = cnn_model.predict(images1, batch_size=32)
features2 = cnn_model.predict(images2, batch_size=32)

# Compute similarity
similarities = compute_similarity(features1, features2)
overall_similarity = np.mean(similarities)

# Print similarity scores
print("\n🔹 *Similarity Scores* 🔹\n")
for i, sim in enumerate(similarities):
    print(f"Image {filenames1[i]} vs {filenames2[i]}: Similarity = {sim:.4f}")
print(f"\n🔹 *Overall Similarity Score*: {overall_similarity:.4f}")

# Visualize results
plot_images(images1, images2, filenames1, filenames2)
plot_similarity(similarities)
plot_feature_differences(features1, features2)
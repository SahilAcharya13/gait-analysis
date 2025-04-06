import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to load and preprocess images from a folder
def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    filenames = sorted(os.listdir(folder))  # Sort to maintain order
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize image
            img = img / 255.0  # Normalize (0-1 range)
            images.append(img)
    return np.array(images), filenames

# Load two different angle datasets
folder1 = "D:/Masters/Sem 2/AISNS/Gait_analysis/GaitDatasetB-silh/001/cl-01/018"  # Change this to your first dataset folder
folder2 = "D:/Masters/Sem 2/AISNS/Gait_analysis/GaitDatasetB-silh/001/cl-01/036"  # Change this to your second dataset folder

images1, filenames1 = load_images_from_folder(folder1)
images2, filenames2 = load_images_from_folder(folder2)

# Ensure both sets have the same number of images
min_len = min(len(images1), len(images2))
images1, images2 = images1[:min_len], images2[:min_len]

# Reshape for CNN input (Adding channel dimension)
images1 = images1.reshape(-1, 128, 128, 1)
images2 = images2.reshape(-1, 128, 128, 1)

# Plot some sample silhouettes
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images1[i].squeeze(), cmap="gray")
    plt.title(f"Angle 1 - {filenames1[i]}")
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(images2[i].squeeze(), cmap="gray")
    plt.title(f"Angle 2 - {filenames2[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Define CNN Model for feature extraction
def create_cnn():
    input_layer = Input(shape=(128, 128, 1))
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
    return Model(input_layer, x)

# Create feature extractor model
cnn_model = create_cnn()

# Extract features
features1 = cnn_model.predict(images1)
features2 = cnn_model.predict(images2)

# Compute similarity (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(features1[i], features2[i]) for i in range(min_len)]

# Compute the overall similarity score
overall_similarity = np.mean(similarities)

# Print similarity scores
print("\n🔹 *Similarity Scores* 🔹\n")
for i, sim in enumerate(similarities):
    print(f"Image {filenames1[i]} vs {filenames2[i]}: Similarity = {sim:.4f}")

# Print overall similarity score
print(f"\n🔹 *Overall Similarity Score*: {overall_similarity:.4f}")

# Plot similarity scores
plt.figure(figsize=(8, 5))
plt.plot(similarities, marker='o', linestyle='-', color="blue", label="Cosine Similarity")
plt.xlabel("Image Index")
plt.ylabel("Similarity Score")
plt.title("Cosine Similarity Between Different Angles")
plt.legend()
plt.grid()
plt.show()

# Visualize feature differences with heatmaps
feature_diff = np.abs(features1 - features2)

plt.figure(figsize=(10, 6))
sns.heatmap(feature_diff[:10], cmap="viridis", annot=False, linewidths=0.5)
plt.title("Feature Differences Between Angles (First 10 Samples)")
plt.xlabel("Feature Index")
plt.ylabel("Image Index")
plt.show()
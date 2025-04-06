import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to silhouette dataset (update this)
dataset_path = "GaitDatasetB-silh/001/nm-02/108"

# Load all silhouette images
silhouettes = []
for filename in sorted(os.listdir(dataset_path)):  # Ensure sorted order
    if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust for your dataset
        img = cv2.imread(os.path.join(dataset_path, filename), cv2.IMREAD_GRAYSCALE)
        silhouettes.append(img)

silhouettes = np.array(silhouettes, dtype=np.float32)  # Convert to NumPy array

# Show one sample silhouette
plt.imshow(silhouettes[5], cmap='gray')
plt.title("Sample Silhouette Frame")
plt.show()

print(f"✅ Loaded {len(silhouettes)} frames from {dataset_path}")

# Compute the Gait Energy Image (GEI)
gei = np.mean(silhouettes, axis=0)

# Normalize values (0-255)
gei = cv2.normalize(gei, None, 0, 255, cv2.NORM_MINMAX)

# Show GEI
plt.imshow(gei, cmap='gray')
plt.title("Gait Energy Image (GEI)")
plt.show()

# Convert GEI to a feature vector
gei_vector = gei.flatten()

print(f"✅ Feature vector shape: {gei_vector.shape}")

from sklearn.metrics.pairwise import cosine_similarity

# Load a second subject's GEI
dataset_path2 = "GaitDatasetB-silh/001/nm-04/126"  # Change path for second subject
silhouettes2 = []
for filename in sorted(os.listdir(dataset_path2)):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(dataset_path2, filename), cv2.IMREAD_GRAYSCALE)
        silhouettes2.append(img)

silhouettes2 = np.array(silhouettes2, dtype=np.float32)
gei2 = np.mean(silhouettes2, axis=0).flatten()

# Show Subject 1 GEI
plt.subplot(1, 2, 1)
plt.imshow(gei, cmap='gray')
plt.title("Subject 1 GEI")

# Show Subject 2 GEI
plt.subplot(1, 2, 2)
plt.imshow(gei2.reshape(gei.shape), cmap='gray')  # Reshape for visualization
plt.title("Subject 2 GEI")

plt.show()

# Compute similarity
similarity = cosine_similarity([gei_vector], [gei2])[0][0]

print(f"✅ Similarity Score: {similarity:.4f}")


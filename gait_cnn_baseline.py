import os
import cv2
import numpy as np

# Path to dataset (adjust as needed)
dataset_path = "D:/Masters/Sem 2/AISNS/Gait_analysis/GaitDatasetB-silh/001"

def load_gait_data(dataset_path):
    images, labels = [], []
    label_dict = {}  # To store unique subject labels

    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(dataset_path):
        # Ignore system files
        files = [f for f in files if not f.startswith('.')]
        
        if len(files) == 0:
            continue  # Skip empty folders

        # Extract subject ID from directory name
        subject_id = os.path.basename(root)
        if subject_id not in label_dict:
            label_dict[subject_id] = len(label_dict)  # Assign unique label

        label = label_dict[subject_id]

        for file in sorted(files):
            img_path = os.path.join(root, file)

            # Read and preprocess image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Skipping unreadable image: {img_path}")
                continue

            img = cv2.resize(img, (64, 64))  # Resize to CNN-compatible size
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

X, y = load_gait_data(dataset_path)
from tensorflow.keras.utils import to_categorical

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=124)  # Change 10 based on the number of subjects
print(f"✅ Updated Labels Shape: {y.shape}")


print(f"✅ Dataset Loaded: {X.shape}, Labels: {y.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN model using `Input(shape=...)`
def build_gait_cnn():
    model = Sequential([
        Input(shape=(64, 64, 1)),  # Corrected input definition
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(124, activation='softmax')  # Adjust based on the number of subjects
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create model
model = build_gait_cnn()
model.summary()


from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CNN
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))
# Save trained model
model.save("gait_ann_model.h5")
print("✅ Model saved successfully as 'gait_cnn_model.h5'")
# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")


# Predict subject identity for a new gait sequence
predictions = model.predict(X_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

print(f"Predicted Labels: {predicted_labels}")


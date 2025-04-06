import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Path to dataset (update as needed)
dataset_path = "D:/Masters/Sem 2/AISNS/Gait_analysis/GaitDatasetB-silh/001/"


# Load and preprocess gait dataset
def load_gait_data(dataset_path):
    images, labels = [], []
    label_dict = {}  # Unique subject labels

    for root, dirs, files in os.walk(dataset_path):
        files = [f for f in files if not f.startswith('.')]  # Ignore system files

        if not files:
            continue  # Skip empty folders

        subject_id = os.path.basename(root)
        if subject_id not in label_dict:
            label_dict[subject_id] = len(label_dict)

        label = label_dict[subject_id]

        for file in sorted(files):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"❌ Skipping unreadable image: {img_path}")
                continue

            img = cv2.resize(img, (64, 64))  # Resize
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, 64, 64, 1).astype('float32') / 255.0  # Normalize
    labels = np.array(labels)
    num_classes = len(np.unique(labels))  # Dynamic class count
    labels = to_categorical(labels, num_classes=num_classes)

    return images, labels, num_classes


# Load dataset
X, y, num_classes = load_gait_data(dataset_path)
print(f"✅ Dataset Loaded: {X.shape}, Labels: {y.shape}")


# Define improved CNN model
def build_improved_gait_cnn(num_classes):
    model = Sequential([
        Input(shape=(64, 64, 1)),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Create model
model = build_improved_gait_cnn(num_classes)
model.summary()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(X_train)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=10,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Save model
model.save("gait_cnn_model.h5")
print("✅ Model saved successfully as 'gait_cnn_model.h5'")

# Predict subject identity for a new gait sequence
predictions = model.predict(X_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

print(f"Predicted Labels: {predicted_labels}")

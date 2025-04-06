import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    print(num_classes)
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

# Early Stopping and Learning Rate Scheduling
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=10,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])

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

# Visualization of Training Results
fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Accuracy', mode='lines+markers'))
fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', mode='lines+markers'))
fig.update_layout(title='Model Accuracy Over Epochs', xaxis_title='Epochs', yaxis_title='Accuracy')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', mode='lines+markers'))
fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss', mode='lines+markers'))
fig.update_layout(title='Model Loss Over Epochs', xaxis_title='Epochs', yaxis_title='Loss')
fig.show()

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

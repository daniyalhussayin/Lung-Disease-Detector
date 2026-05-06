import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# === CONFIG ===
DATADIR = "dataset"
CATEGORIES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 100

print("📥 Loading images...")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category, "images")  # e.g., dataset/NORMAL/images
    class_num = CATEGORIES.index(category)
    try:
        for img_name in os.listdir(path)[:300]:  # First 300 images from each class
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is not None:
                    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append(resized)
                    labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    except FileNotFoundError:
        print(f"❌ Folder not found: {path}")

# === PREPARE DATA ===
X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
y = to_categorical(labels, num_classes=len(CATEGORIES))        # One-hot encoding

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === CNN MODEL ===
print("🧠 Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# === EVALUATE ===
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# === SAVE MODEL ===
model.save("lung_disease_model.h5")
print("💾 Model saved as lung_disease_model.h5")

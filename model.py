import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 128  # resize all images to this size

birdA_path = "spectrograms_binary/asbfly"
birdB_path = "spectrograms_binary/brnhao"

# ----------------------------
# LOAD IMAGES
# ----------------------------
def load_images(folder, label):
    data = []
    labels = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten()  # convert to 1D

        data.append(img)
        labels.append(label)

    return data, labels


print("📂 Loading data...")

X1, y1 = load_images(birdA_path, 0)  # Bird A
X2, y2 = load_images(birdB_path, 1)  # Bird B

X = np.array(X1 + X2)
y = np.array(y1 + y2)

print("✅ Total samples:", len(X))

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# NORMALIZATION
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 2-LAYER NEURAL NETWORK
# ----------------------------
print("🧠 Training model...")

model = MLPClassifier(
    hidden_layer_sizes=(128,),  # 1 hidden layer
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
print("📊 Evaluating...")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 Accuracy:", acc)
print("📌 Confusion Matrix:\n", cm)

# ----------------------------
# TEST ON SINGLE IMAGE
# ----------------------------
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten()

    img = scaler.transform([img])

    pred = model.predict(img)[0]

    if pred == 0:
        print("🟣 Prediction: asbfly")
    else:
        print("🟠 Prediction: brnhao")


# Example test (change path if needed)
# predict_image("spectrograms_binary/asbfly/sample.png")
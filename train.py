import tensorflow as tf
from tensorflow.keras import layers, models
import os

# path to split data
DATA_DIR = "data_split"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ======================
# LOAD DATA
# ======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ SAVE CLASS NAMES BEFORE MAP
class_names = train_ds.class_names
print("Classes:", class_names)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ======================
# NORMALIZATION
# ======================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# ======================
# MODEL
# ======================
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# ======================
# COMPILE
# ======================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# TRAIN
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ======================
# TEST
# ======================
test_loss, test_acc = model.evaluate(test_ds)

print("🔥 Test Accuracy:", test_acc)
model.save("bird_model.h5")
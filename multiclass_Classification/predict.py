import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# load model
model = tf.keras.models.load_model("bird_model.h5")

# IMPORTANT: same class order as training
class_names = ['Barfly', 'asbfly', 'ashpri', 'bkcbul']

# load image
img_path = "XC134896 (1)_chunk0.png"   

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Prediction:", predicted_class)
print("Confidence:", confidence)
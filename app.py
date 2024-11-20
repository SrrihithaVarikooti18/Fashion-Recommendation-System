# app.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Path to dataset
data_dir = 'C:/Users/MANASWINI KARNATAKA/Desktop/data'
filenames = []
feature_list = []

# Process each image and extract features
for category in ['apparel', 'footwear']:
    category_path = os.path.join(data_dir, category)
    for subdir, _, files in os.walk(category_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(subdir, file)
                filenames.append(img_path)
                feature_list.append(extract_features(img_path, model))

# Save features and filenames for later use
pickle.dump(feature_list, open('featurevector.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
print("Feature extraction completed and saved.")

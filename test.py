# test.py
import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Load saved features and filenames
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Load a test image
img_path = 'C:/Users/MANASWINI KARNATAKA/Desktop/FinalMini/2746.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
query_features = model.predict(preprocessed_img).flatten()
query_features = query_features / norm(query_features)

# Nearest neighbor search
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([query_features])

# Display similar images
for file in indices[0][1:6]:
    similar_img = cv2.imread(filenames[file])
    cv2.imshow('Similar Image', cv2.resize(similar_img, (512, 512)))
    cv2.waitKey(0)
cv2.destroyAllWindows()

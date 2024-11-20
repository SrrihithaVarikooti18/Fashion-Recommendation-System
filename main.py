import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import os

# Initialize the ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Load the feature vectors and filenames
feature_list = np.array(pickle.load(open('featurevector.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Streamlit UI configuration
st.set_page_config(page_title="Fashion Recommender System", layout="wide")
st.title("Fashion Recommender System")

# Sidebar for recommendation method selection
st.sidebar.header("Choose a Recommendation Method")
recommendation_method = st.sidebar.selectbox(
    "Recommendation Method",
    ["Content-based", "Collaborative Filtering", "Hybrid"]
)

# File upload section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Initialize wishlist
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []
    st.session_state.wishlist_images = []

# Function for mock collaborative filtering
def mock_collaborative_filtering():
    return np.random.choice(len(filenames), 6, replace=False).reshape(1, -1)

# Function for hybrid recommendation
def combine_hybrid_results(content_indices, collab_indices):
    hybrid_indices = []
    for i in range(1, 6):
        if i % 2 == 0:
            hybrid_indices.append(content_indices[0][i])
        else:
            hybrid_indices.append(collab_indices[0][i])
    return [hybrid_indices]

# Process the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.success("Image uploaded successfully!")

    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Nearest neighbor search for content-based filtering
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    if recommendation_method == "Content-based":
        distances, indices = neighbors.kneighbors([normalized_result])
    elif recommendation_method == "Collaborative Filtering":
        indices = mock_collaborative_filtering()
    elif recommendation_method == "Hybrid":
        content_indices = neighbors.kneighbors([normalized_result])[1]
        collab_indices = mock_collaborative_filtering()
        indices = combine_hybrid_results(content_indices, collab_indices)

    # Display recommendations
    st.write("Here are your top fashion recommendations:")
    cols = st.columns(5)
    for col, i in zip(cols, indices[0][1:6]):
        recommended_img_path = filenames[i]
        recommended_img = cv2.imread(recommended_img_path)
        recommended_img = cv2.cvtColor(recommended_img, cv2.COLOR_BGR2RGB)
        with col:
            st.image(recommended_img, use_column_width=True, caption=os.path.basename(recommended_img_path))
            if st.checkbox(f"Add to Wishlist {os.path.basename(recommended_img_path)}", key=f"checkbox_{i}"):
                if os.path.basename(recommended_img_path) not in st.session_state.wishlist:
                    st.session_state.wishlist.append(os.path.basename(recommended_img_path))
                    st.session_state.wishlist_images.append(recommended_img)
                    st.success(f"{os.path.basename(recommended_img_path)} added to wishlist!")

# Display wishlist
with st.expander("Your Wishlist", expanded=False):
    if st.session_state.wishlist:
        for idx, item in enumerate(st.session_state.wishlist):
            if st.checkbox(f"Show {item}", key=f"wishlist_checkbox_{idx}"):
                st.image(st.session_state.wishlist_images[idx], caption=item, use_column_width=True)
    else:
        st.write("Your Wishlist is empty.")
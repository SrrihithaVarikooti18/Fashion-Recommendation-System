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

# Main content area
st.title("Fashion Recommender System")
st.markdown("""
    <style>
        .main-banner {
            text-align: center;
            font-size: 24px;
            margin-top: 50px;
        }
        .upload-area {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.header("Shop By Department")

# Men's Fashion Section
with st.sidebar.expander("Men's Fashion", expanded=True):
    st.markdown("- T-shirts")
    st.markdown("- Shirts")
    st.markdown("- Jeans")

# Women's Fashion Section
with st.sidebar.expander("Women's Fashion", expanded=True):
    st.markdown("- Dresses")
    st.markdown("- Tops")
    st.markdown("- Skirts")

# Accessories Section
with st.sidebar.expander("Accessories", expanded=True):
    st.markdown("- Watches")
    st.markdown("- Bags and Luggages")
    st.markdown("- Sunglasses")

# Stores Section
with st.sidebar.expander("Stores", expanded=True):
    st.markdown("- Sportswear")
    st.markdown("- The Designer Boutique")
    st.markdown("- Fashion sales and deals")

# Define the path to your image file
image_path = 'C:/Users/MANASWINI KARNATAKA/Downloads/projectLogo.jpg'  # Update this path accordingly

if os.path.isfile(image_path):
    # Load the image using OpenCV
    check_img = cv2.imread(image_path)
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    
    # Resize the image (e.g., reducing height to 250 pixels, preserving aspect ratio)
    desired_height = 250  # Adjust this value to decrease the height
    aspect_ratio = check_img.shape[1] / check_img.shape[0]  # width / height
    new_width = int(desired_height * aspect_ratio)

    # Resize the image to the new dimensions
    resized_check_img = cv2.resize(check_img, (new_width, desired_height))

    # Display the resized image
    st.image(resized_check_img, caption="Trends in Fashion Ecommerce", use_column_width=True)
else:
    st.warning("Banner image not found. Please check the path or place 'check.png' in the correct directory.")


# File upload section
st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# Initialize wishlist
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []
    st.session_state.wishlist_images = []  # Store images for the wishlist

# Process the uploaded image and show recommendations
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)  # Adjust the width of the input image as needed
    st.success("Image uploaded successfully!")

    # Load the uploaded image for feature extraction
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Perform nearest neighbor search
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([normalized_result])

    # Display the top recommendations
    st.write("Here are your top fashion recommendations:")
    cols = st.columns(5)  # Create 5 columns for 5 images

    for col, i in zip(cols, indices[0][1:6]):
        recommended_img_path = filenames[i]
        recommended_img = cv2.imread(recommended_img_path)
        recommended_img = cv2.cvtColor(recommended_img, cv2.COLOR_BGR2RGB)
        with col:
            st.image(recommended_img, use_column_width=True, caption=os.path.basename(recommended_img_path))
            # Add checkbox for adding to wishlist
            if st.checkbox(f"Add to Wishlist {os.path.basename(recommended_img_path)}", key=f"checkbox_{i}"):
                if os.path.basename(recommended_img_path) not in st.session_state.wishlist:
                    st.session_state.wishlist.append(os.path.basename(recommended_img_path))
                    st.session_state.wishlist_images.append(recommended_img)  # Store the image for display
                    st.success(f"{os.path.basename(recommended_img_path)} added to wishlist!")

# Display wishlist in a box
with st.expander("Your Wishlist", expanded=False):
    if st.session_state.wishlist:
        for idx, item in enumerate(st.session_state.wishlist):
            # Use a checkbox to select an item
            if st.checkbox(f"Show {item}", key=f"wishlist_checkbox_{idx}"):
                st.image(st.session_state.wishlist_images[idx], caption=item, use_column_width=True)  # Display the image when checked
    else:
        st.write("Your Wishlist is empty.")

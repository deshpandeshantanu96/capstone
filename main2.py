import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load text-based embeddings (product names with image filenames)
with open("product_embeddings_with_images.pkl", "rb") as f:
    embeddings_df = pickle.load(f)

# Load image-based embeddings and filenames for image recommendation
with open("embeddings.pkl", "rb") as f:
    feature_list = np.array(pickle.load(f))
    feature_list = feature_list.reshape(feature_list.shape[0], -1)
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Initialize the ResNet model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

# Load SentenceTransformer model for NLP-based search
model_nlp = SentenceTransformer('all-MiniLM-L6-v2')


# Define functions
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(img_array, axis=0)
    preprocessed_image = preprocess_input(expanded_image_array)
    result = model.predict(preprocessed_image)
    return result.flatten() / norm(result)


def recommend_image(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices, distances


# Streamlit UI
st.title("Fashion Recommender System")

# Text search input
search_term = st.text_input("Search for products")

# Image upload input
uploaded_file = st.file_uploader("Or upload an image to find similar products:", type=["jpg", "jpeg", "png"])

# Check if text search is performed
if search_term and not uploaded_file:
    # Convert search term to embedding
    search_vector = model_nlp.encode(search_term)

    # Calculate cosine similarity with each product embedding
    embeddings_df['similarity'] = embeddings_df['embedding'].apply(
        lambda x: np.dot(search_vector, x) / (norm(search_vector) * norm(x))
    )

    # Sort results by similarity
    top_results = embeddings_df.sort_values(by="similarity", ascending=False).head(5)

    # Print similarity scores for evaluation
    print("\nText-based Recommendations and Similarity Scores:")
    for idx, row in top_results.iterrows():
        print(f"Product: {row['productDisplayName']} - Similarity: {row['similarity']:.4f}")

    # Display recommended items without similarity scores
    st.write("Recommendations based on search:")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if idx < len(top_results):
            img_path = "images/" + str(top_results.iloc[idx]["id"]) + ".jpg"
            product_name = top_results.iloc[idx].get('productDisplayName', 'Unknown Product')
            col.image(img_path, width=100)
            col.write(product_name)

    # Extract cosine similarity values
    cosine_similarities = embeddings_df['similarity'].values

    # Plot histogram of cosine similarities
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cosine_similarities, bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f"Cosine Similarity Distribution for Search Term: '{search_term}'")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")

    # Show histogram in Streamlit
    st.pyplot(fig)

# Check if image upload is performed
elif uploaded_file and not search_term:
    st.session_state.last_state = None
    # Save the uploaded file temporarily
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    display_image = Image.open(uploaded_file)
    st.image(display_image, caption="Uploaded Image", width=200)

    # Extract features from the uploaded image and find recommendations
    features = feature_extraction(file_path, model)
    indices, distances = recommend_image(features, feature_list)

    # Print Euclidean distance scores for evaluation
    print("\nImage-based Recommendations and Distance Scores:")
    for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
        img_path = filenames[index]
        numeric_id = int(''.join(filter(str.isdigit, img_path)))
        product_name = embeddings_df.loc[embeddings_df['id'] == numeric_id, 'productDisplayName'].values[0]
        print(f"Product: {product_name} - Distance: {distance:.4f}")

    # Display recommended items without distance scores
    st.write("Recommendations based on uploaded image:")
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        if idx < len(indices[0]):  # Ensure we don't exceed the number of recommendations
            img_path = filenames[indices[0][idx]]
            col.image(img_path, width=100)
            numeric_id = int(''.join(filter(str.isdigit, img_path)))
            product_name = embeddings_df.loc[embeddings_df['id'] == numeric_id, 'productDisplayName'].values[0]
            col.write(product_name)

    # Plot all distances between uploaded image and dataset images
    all_distances = np.linalg.norm(feature_list - features, axis=1)
    print("\nAll Distances from the Uploaded Image to Dataset Images:")


    # Plot the distribution of all distances
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Euclidean Distances from Uploaded Image to Dataset Images")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.show()



from flask import Flask, request, jsonify
import faiss
import numpy as np
from PIL import Image
import os

import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)


# Set up Faiss index (this is just a simple example)
def setup_faiss_index():
    # Define the dimensionality of your image vectors
    dimension = 2048  # For example, if your image features are 2048-dimensional (e.g., extracted from a CNN)

    # Initialize the index
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)

    # Example: Add some sample image features to the index
    sample_image_features = [
        # Example image features (replace with your own)
        np.random.rand(dimension),  # Image feature vector 1
        np.random.rand(dimension),  # Image feature vector 2
        # Add more image feature vectors as needed
    ]

    # Convert sample image features to numpy array
    sample_image_features_np = np.array(sample_image_features, dtype=np.float32)

    # Add image features to the index
    index.add(sample_image_features_np)

    # Return the initialized index
    return index


# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Load a specific image
    image_path = '../image/image.png'
    from PIL import Image

    # Open the image
    image = Image.open(image_path)

    # Convert RGBA image to RGB
    image = image.convert('RGB')

    # Resize the image to the expected input size of ResNet50 (e.g., 224x224)
    image = image.resize((512, 512))


    image_features = extract_features(image)

    search_results = faiss_index.search(image_features,5)
    search_results_list = [result.tolist() for result in search_results]
    # Return search results (replace with your response)
    return jsonify({'results': search_results_list})

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define a function to extract features from an image using the ResNet50 model
def extract_features(image_data):
    # Preprocess the image for the ResNet50 model
    img = keras_image.img_to_array(image_data)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features from the image using the ResNet50 model
    features = resnet_model.predict(img)
    print(features)
    return features

if __name__ == '__main__':
    faiss_index = setup_faiss_index()  # Initialize Faiss index
    query_vector = np.random.rand(1, 2048).astype(np.float32)  # Arama sorgusu vektörünü buraya yerleştirin

    # Arama yap
    k = 5  # Toplamda döndürülecek sonuç sayısı
    distances, indices = faiss_index.search(query_vector, k)

    # Arama sonuçlarını göster
    for i in range(len(indices)):
        print(f"Arama Sonucu {i + 1}:")
        for j in range(len(indices[i])):
            doc_id = indices[i][j]  # Belgelerin kimliklerini al
            distance = distances[i][j]  # Benzerlik mesafesini al
            print(f"Belge Kimliği: {doc_id}, Benzerlik Mesafesi: {distance}")
        print("\n")


    app.run(debug=True)
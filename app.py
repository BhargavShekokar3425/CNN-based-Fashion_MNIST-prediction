# making an end to end web app using streamlit

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the saved model
model = load_model("fashion_mnist_cnn.h5")

# Define the class labels for Fashion MNIST
label_names = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat", 
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot"
]

# App title and description
st.title("🧥 Fashion MNIST Predictor")
st.markdown("""
Welcome to the Fashion MNIST Predictor app! 🎨  
Upload a grayscale image of a fashion item, and the model will predict the category.  
Supported categories:  
**T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot**
""")

# Image upload
uploaded_file = st.file_uploader("Upload an image (28x28 grayscale PNG or JPEG)", type=["png", "jpeg", "jpg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Process the image
    try:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = img_to_array(image) / 255.0  # Normalize
        image_array = image_array.reshape(1, 28, 28, 1)  # Add batch dimension

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Map predicted class to class name
        predicted_class_name = label_names[predicted_class]

        # Display the results
        st.subheader("Prediction Results:")
        st.write(f"**Predicted Class**: {predicted_class_name} (Class ID: {predicted_class})")
        st.write(f"**Confidence**: {confidence * 100:.2f}%")
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an image to get started.")



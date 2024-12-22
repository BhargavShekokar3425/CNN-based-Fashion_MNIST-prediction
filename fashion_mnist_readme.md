
# Fashion MNIST Model with Sample Prediction and Streamlit Web App

## Overview

This project demonstrates a Fashion MNIST classification model using TensorFlow and Keras. The model is trained on the Fashion MNIST dataset, which consists of grayscale images of 10 fashion categories. The project includes:

- A trained model for classifying images.
- A sample image prediction script to make predictions with the model.
- A Streamlit web application for interactive image classification.

## Setup

### 1. Install the required libraries:

To get started, ensure you have the necessary Python libraries installed. You can install the required libraries using pip:

```bash
pip install tensorflow keras opencv-python matplotlib numpy streamlit
```

### 2. Download the Fashion MNIST dataset:
This dataset is automatically downloaded using Keras when you load the data, so no manual setup is required.

### 3. Load the Model:

You can load the pre-trained model `fashion_mnist_model.h5` using the following code:

```python
from tensorflow.keras.models import load_model

model = load_model("fashion_mnist_model.h5")
```

Make sure the model file is available in the current directory.

## Predicting a Sample Image

To make predictions with the trained model, you can use the following script:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("fashion_mnist_model.h5")

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

# Preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Make a prediction
def predict_sample(model, image_path, label_names):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    return label_names[predicted_label], confidence

# Sample image path
sample_image_path = "sample_image.jpg"

# Predict
predicted_class, confidence = predict_sample(model, sample_image_path, label_names)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
```

### Sample Output:

```
Predicted Class: Sneaker, Confidence: 0.98
```

## Streamlit Web App

To make the model accessible via a web interface, we use Streamlit. Here's how to set up the web app:

### 1. Create a Streamlit Web App:

Create a new file `app.py` and use the following code:

```python
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("fashion_mnist_model.h5")

# Define class labels
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

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Function to predict the class of an image
def predict_class(model, image, label_names):
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    return label_names[predicted_label], confidence

# Streamlit web app layout
st.title("Fashion MNIST Classifier")
st.write("Upload an image of a fashion item for classification")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Display the image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the uploaded image to a NumPy array
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    
    # Predict the class
    predicted_class, confidence = predict_class(model, image, label_names)
    
    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
```

### 2. Run the Streamlit Web App:

To run the Streamlit app, use the following command in your terminal:

```bash
streamlit run app.py
```

This will start a web server, and you can interact with the app by uploading an image of a fashion item for classification.

## Conclusion

This project demonstrates a Fashion MNIST classification model with the ability to predict fashion item categories. It also provides an interactive web app interface for users to upload images and get predictions.

Enjoy experimenting with the model and web app!

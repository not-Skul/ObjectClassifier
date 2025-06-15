import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_cifar.h5")

model = load_model()
class_names = ["airplane", "automible","bird","cat","deer","dog","frog","horse","ship","truck"]  # Edit based on your model classes

# UI
st.title("Object classifier")
st.write("Upload an image, This model can classify CIFAR-10 images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((32, 32))             # Resize as per model input
    img_array = np.array(img) / 255.0          # Normalize
    img_array = np.expand_dims(img_array, 0)   # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: *{predicted_class}* ({confidence:.2f}% confidence)")
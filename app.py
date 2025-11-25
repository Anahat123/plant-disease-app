import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import traceback

st.title("Plant Disease Detection App 🌱")

MODEL_FILE = "model.keras"

@st.cache_resource
def load_model_only_keras():
    try:
        # compile=False avoids version conflicts
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        return model
    except Exception as e:
        st.error("Failed to load .keras model.")
        st.text(traceback.format_exc())
        raise e

# Load .keras model
try:
    model = load_model_only_keras()
    st.success("Model loaded successfully!")
except:
    st.stop()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    try:
        preds = model.predict(arr)
        class_idx = int(np.argmax(preds))
        st.success(f"Predicted class: {class_idx}")
    except Exception as e:
        st.error("Prediction failed.")
        st.text(traceback.format_exc())

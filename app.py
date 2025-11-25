# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import traceback
import os

st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("Plant Disease Detection App 🌱")

MODEL_FILE = "model.keras"
CLASS_JSON_PATH = "class_names.json"  # <-- using your uploaded file path

@st.cache_data
def load_class_names(path=CLASS_JSON_PATH):
    if not os.path.exists(path):
        st.error(f"Class names file not found at: {path}")
        return []
    try:
        with open(path, "r") as f:
            names = json.load(f)
        return names
    except Exception:
        st.error("Failed to load class names JSON.")
        st.text(traceback.format_exc())
        return []

@st.cache_resource
def load_model_only_keras(path=MODEL_FILE):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        raise FileNotFoundError(path)
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception:
        st.error("Failed to load .keras model.")
        st.text(traceback.format_exc())
        raise

# load resources
CLASS_NAMES = load_class_names()
try:
    model = load_model_only_keras()
    st.success("Model loaded successfully!")
except Exception:
    st.stop()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    try:
        preds = model.predict(arr)
        probs = np.squeeze(preds)
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        if CLASS_NAMES and idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[idx]
        else:
            class_name = f"Class_{idx}"

        st.success(f"Predicted: **{class_name}**  —  Confidence: **{prob:.2%}**")
    except Exception:
        st.error("Prediction failed.")
        st.text(traceback.format_exc())


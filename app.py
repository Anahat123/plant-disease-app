import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import traceback
from tensorflow.keras.applications.efficientnet import preprocess_input   # 🟢 IMPORTANT FIX

st.title("🌿 Plant Disease Detection App (EfficientNetB0)")

MODEL_FILE = "model.keras"

# ------------------- CLASS NAMES (aligned perfectly) -------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
   "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ------------------- Load Model -------------------
@st.cache_resource
def load_model_only_keras():
    try:
        return tf.keras.models.load_model(MODEL_FILE, compile=False)
    except Exception:
        st.error("❌ Failed to load .keras model")
        st.text(traceback.format_exc())
        raise

model = load_model_only_keras()
st.success("✅ Model Loaded Successfully!")

# ------------------- Image Upload + Prediction -------------------
uploaded_file = st.file_uploader("Upload a leaf image 🌿", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))   # matches training size
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 🟢 EfficientNetB0 correct preprocessing
    arr = np.array(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)   # <--- The fix that improves prediction!

    try:
        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        class_name = CLASS_NAMES[idx]

        st.subheader("🔍 Prediction Result")
        st.success(f"🌱 Disease: **{class_name}**")
        st.write(f"📊 Confidence: **{confidence:.2%}**")

    except Exception:
        st.error("❌ Prediction Failed")
        st.text(traceback.format_exc())

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ===========================================================
# 1) Load Model
# ===========================================================
@st.cache_resource  # prevents the model from reloading at every interaction
def load_model():
    model = tf.keras.models.load_model("plant_disease_effb0_best.keras")
    return model

model = load_model()

# ===========================================================
# 2) Class Labels (same order as during training)
# ===========================================================
class_names = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy', 
'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight',
'Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
'Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# ===========================================================
# 3) Streamlit UI
# ===========================================================
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection using EfficientNetB0")
st.write("Upload a plant leaf image to identify disease with high accuracy.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ===========================================================
# 4) Prediction Function
# ===========================================================
def preprocess_image(img):
    img = img.resize((224,224))
    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img) 
    img = np.expand_dims(img, axis=0)  # (1,224,224,3)
    return img

def predict(image):
    tensor = preprocess_image(image)
    preds = model.predict(tensor)
    index = np.argmax(preds)
    confidence = np.max(preds) * 100
    return class_names[index], confidence

# ===========================================================
# 5) Display Results
# ===========================================================
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", width=350)

    with st.spinner("Analyzing image... 🔍"):
        label, confidence = predict(img)

    st.success(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # If healthy/ not healthy quick indicator
    if "healthy" in label.lower():
        st.markdown("### 🌱 Status: Healthy ✓")
    else:
        st.markdown("### ⚠ Disease Detected")

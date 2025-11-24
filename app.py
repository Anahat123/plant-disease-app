# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

st.set_page_config(layout="centered", page_title="Plant Disease Classifier")

MODEL_PATH = "plant_disease_effb0_final.keras"   # place your saved model here
CLASS_JSON = "class_names.json"                  # save your class names as JSON

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(path=MODEL_PATH):
    model = tf.keras.models.load_model(path)
    return model

@st.cache(allow_output_mutation=True)
def load_class_names(path=CLASS_JSON):
    try:
        with open(path, "r") as f:
            names = json.load(f)
    except Exception:
        # fallback to numeric labels if file missing
        names = [f"class_{i}" for i in range(100)]
    return names

IMG_SIZE = (224,224)   # must match training

def preprocess_image_pil(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    return img

def top_k_preds(probs, k=5):
    idx = np.argsort(probs)[::-1][:k]
    return [(idx[i], float(probs[idx[i]])) for i in range(len(idx))]

def gradcam_overlay(model, img_np, class_idx):
    # create small gradcam
    img_input = np.expand_dims(preprocess_image_pil(Image.fromarray(img_np)), axis=0)
    # pick a conv layer near the top; try to find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv = layer.name
            break
    if last_conv is None:
        return img_np  # fallback
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_input)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0].numpy()
    conv_outputs = conv_outputs[0].numpy()
    weights = np.mean(grads, axis=(0,1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    if cam.max() == 0:
        return img_np
    cam = cam / cam.max()
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cm.jet(cam)[:, :, :3] * 255.0
    overlay = 0.5 * heatmap + 0.5 * cv2.resize(img_np, IMG_SIZE)
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay

# UI
st.title("Plant Disease Classifier (EfficientNetB0)")
st.write("Upload a leaf image — the model will predict disease and show Grad-CAM.")

model = load_model()
class_names = load_class_names()
num_classes = model.output_shape[-1]

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    img_np = np.array(img.convert("RGB"))
    x = preprocess_image_pil(img)
    preds = model.predict(np.expand_dims(x,0))[0]
    top5 = top_k_preds(preds, k=5)
    st.markdown("### Top predictions")
    for i, (idx, prob) in enumerate(top5):
        label = class_names[idx] if idx < len(class_names) else str(idx)
        st.write(f"{i+1}. **{label}** — {prob:.4f}")
    # Grad-CAM overlay
    top_idx = int(top5[0][0])
    overlay = gradcam_overlay(model, img_np, top_idx)
    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

# app.py
import os
import traceback
import json
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

st.set_page_config(layout="centered", page_title="Plant Disease Classifier (EffB0)")

# --- Paths (files you generated in Kaggle) ---
MODEL_ARCH_PATH = "model_arch.json"
MODEL_WEIGHTS_PATH = "model_weights.weights.h5"
CLASS_JSON = "class_names.json"

IMG_SIZE = (224, 224)

# --- Helpers ---
def safe_load_class_names(path=CLASS_JSON):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # fallback list if class file missing
        return [f"class_{i}" for i in range(100)]

@st.cache_resource
def load_model_from_json(arch_path=MODEL_ARCH_PATH, weights_path=MODEL_WEIGHTS_PATH):
    """Load architecture JSON + weights. Shows clear error info on failure."""
    st.write("Loading model architecture from:", arch_path)
    st.write("Loading model weights from:", weights_path)

    if not os.path.exists(arch_path):
        raise FileNotFoundError(f"Model architecture file not found: {arch_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    try:
        # Import TF lazily to avoid import-time issues blocking the UI
        import tensorflow as tf
        from tensorflow.keras.models import model_from_json
    except Exception as e:
        st.error("Failed to import TensorFlow in the runtime.")
        st.text(repr(e))
        st.text(traceback.format_exc())
        raise

    try:
        with open(arch_path, "r") as f:
            arch = f.read()
        model = model_from_json(arch)
        model.load_weights(weights_path)
        st.write("✅ Model loaded (architecture + weights).")
        return model
    except Exception as e:
        st.error("Model load failed — see full traceback below.")
        st.text(repr(e))
        st.text(traceback.format_exc())
        raise

def preprocess_image_pil(pil_img):
    """Return preprocessed image ready for model.predict (no batch dim)."""
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    # use EfficientNet preprocessing if available, otherwise simple scaling
    try:
        import tensorflow as tf
        pre = tf.keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    except Exception:
        pre = img.astype("float32") / 255.0
    return pre

def top_k_preds(probs, k=5):
    idx = np.argsort(probs)[::-1][:k]
    return [(int(idx[i]), float(probs[idx[i]])) for i in range(len(idx))]

def gradcam_overlay(model, img_np, class_idx):
    """Compute Grad-CAM overlay on top of RGB numpy image (H,W,3)."""
    import tensorflow as tf
    # prepare input
    x = np.expand_dims(preprocess_image_pil(Image.fromarray(img_np)), axis=0)

    # find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        # some layers have no output_shape attribute; guard with getattr
        out_shape = getattr(layer, "output_shape", None)
        if out_shape and isinstance(out_shape, tuple) and len(out_shape) == 4:
            last_conv = layer.name
            break
    if last_conv is None:
        return img_np

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return img_np
    grads = grads[0].numpy()
    conv_outputs = conv_outputs[0].numpy()
    # global average pooling on gradients
    weights = np.mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]
    cam = np.maximum(cam, 0)
    if np.max(cam) == 0:
        return img_np
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cm.jet(cam)[:, :, :3] * 255.0
    overlay = 0.5 * heatmap + 0.5 * cv2.resize(img_np, IMG_SIZE)
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay

# --- Load resources ---
try:
    model = load_model_from_json()
except Exception as e:
    st.stop()  # stop execution; the UI already shows error details

class_names = safe_load_class_names()
num_classes = getattr(model, "output_shape", (None, None))[-1]

# --- UI ---
st.title("Plant Disease Classifier (EfficientNetB0)")
st.write("Upload a leaf image — the model will predict disease and show Grad-CAM.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    try:
        img = Image.open(uploaded)
    except Exception:
        st.error("Unable to open the uploaded image. Upload a different JPG/PNG.")
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)
    img_np = np.array(img.convert("RGB"))

    # Preprocess and predict
    x = preprocess_image_pil(img)
    x_batch = np.expand_dims(x, axis=0)

    try:
        preds = model.predict(x_batch)[0]
    except Exception as e:
        st.error("Model prediction failed. See traceback:")
        st.text(repr(e))
        st.text(traceback.format_exc())
        st.stop()

    top5 = top_k_preds(preds, k=min(5, len(preds)))
    st.markdown("### Top predictions")
    for i, (idx, prob) in enumerate(top5):
        label = class_names[idx] if idx < len(class_names) else str(idx)
        st.write(f"{i+1}. **{label}** — {prob:.4f}")

    top_idx = int(top5[0][0])
    try:
        overlay = gradcam_overlay(model, img_np, top_idx)
        st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
    except Exception as e:
        st.warning("Grad-CAM failed; showing original image instead.")
        st.text(repr(e))
        st.image(img_np, caption="Original image", use_column_width=True)

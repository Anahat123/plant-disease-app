# app.py - Enhanced Streamlit UI for Plant Disease Classifier
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
import os
from io import BytesIO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

st.set_page_config(layout="wide", page_title="Plant Disease Classifier - Enhanced")

# --- CONFIG - update filenames if needed
MODEL_PATH = "plant_disease_effb0_best.keras"   # keep in same folder
CLASS_JSON = "class_names.json"
IMG_SIZE = (224, 224)

# --- Helpers
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
        names = [f"class_{i}" for i in range(100)]
    return names

def preprocess_image_pil(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    return img

def predict_np(model, img_np):
    x = preprocess_image_pil(Image.fromarray(img_np))
    preds = model.predict(np.expand_dims(x, 0))[0]
    return preds

def top_k_from_probs(probs, class_names, k=5, threshold=0.0):
    idx = np.argsort(probs)[::-1]
    result = []
    for i in idx:
        if probs[i] < threshold:
            break
        result.append((class_names[i], float(probs[i])))
        if len(result) >= k:
            break
    return result

def gradcam_overlay(model, img_np, class_idx):
    # simple Grad-CAM using last conv layer found
    img_input = np.expand_dims(preprocess_image_pil(Image.fromarray(img_np)), axis=0)
    last_conv = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv = layer.name
            break
    if last_conv is None:
        return img_np
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_input)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0].numpy()
    conv_outputs = conv_outputs[0].numpy()
    weights = np.mean(grads, axis=(0,1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:,:,i]
    cam = np.maximum(cam, 0)
    if cam.max() == 0:
        return img_np
    cam = cam / cam.max()
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = cm.jet(cam)[:,:,:3] * 255.0
    overlay = 0.5 * heatmap + 0.5 * cv2.resize(img_np, IMG_SIZE)
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay

# --- Load model & classes
model = load_model()
class_names = load_class_names()
NUM_CLASSES = len(class_names)

# --- Sidebar controls
st.sidebar.title("Controls")
mode = st.sidebar.selectbox("Mode", ["Single Image", "Batch Folder", "Demo Gallery", "Eval Folder (labeled)"])
top_k = st.sidebar.slider("Top-K predictions", min_value=1, max_value=10, value=5)
threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay (slower)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Model: **{MODEL_PATH}**")
st.sidebar.markdown(f"Classes detected: **{NUM_CLASSES}**")

# --- Top bar
st.title("Plant Disease Classifier — Enhanced Interface")
st.markdown("Upload images, run batch inference, or evaluate a labeled dataset. Visualize predictions using Grad-CAM.")

# --- Main layout
left_col, right_col = st.columns([2,1])

# Single Image UI
if mode == "Single Image":
    with left_col:
        uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            img_np = np.array(image)
            st.image(image, caption="Uploaded image", use_column_width=True)
            # predict
            probs = predict_np(model, img_np)
            topk = top_k_from_probs(probs, class_names, k=top_k, threshold=threshold)
            # show list
            st.markdown("### Top predictions")
            for i,(lab,p) in enumerate(topk):
                st.write(f"{i+1}. **{lab}** — {p:.4f}")
            # Grad-CAM
            if show_gradcam and len(topk)>0:
                with st.spinner("Computing Grad-CAM..."):
                    overlay = gradcam_overlay(model, img_np, class_names.index(topk[0][0]))
                    st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)
    with right_col:
        st.markdown("#### Prediction Confidence")
        if uploaded:
            df = pd.DataFrame({"label":[t[0] for t in topk], "prob":[t[1] for t in topk]})
            st.bar_chart(df.set_index("label")["prob"])

# Batch Folder UI
elif mode == "Batch Folder":
    st.info("Provide the folder name containing images. Example: `batch_images/`")
    folder_name = st.text_input("Folder name (relative to this app)", value="batch_images")
    run_btn = st.button("Run batch inference")
    if run_btn:
        img_files = []
        for root,_,files in os.walk(folder_name):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    img_files.append(os.path.join(root,f))
        st.write(f"Found {len(img_files)} images. Running predictions...")
        results = []
        progress = st.progress(0)
        for i,fp in enumerate(img_files):
            try:
                pil = Image.open(fp).convert("RGB")
                probs = predict_np(model, np.array(pil))
                topk = top_k_from_probs(probs, class_names, k=top_k, threshold=threshold)
                best_label = topk[0][0] if topk else ""
                best_prob = topk[0][1] if topk else 0.0
                results.append({"file":fp, "prediction":best_label, "probability":best_prob})
            except Exception as e:
                results.append({"file":fp, "prediction":"ERROR", "probability":0.0})
            progress.progress((i+1)/len(img_files))
        df = pd.DataFrame(results)
        st.dataframe(df.head(200))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv, "batch_predictions.csv")

# Demo gallery
elif mode == "Demo Gallery":
    demo_folder = "demo_images"
    if os.path.exists(demo_folder):
        st.markdown("### Demo Images")
        cols = st.columns(4)
        files = [os.path.join(demo_folder,f) for f in os.listdir(demo_folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        for i,fp in enumerate(files):
            with cols[i % 4]:
                img = Image.open(fp).convert("RGB")
                st.image(img, use_column_width=True)
                if st.button(f"Predict #{i+1}", key=f"pred{i}"):
                    probs = predict_np(model, np.array(img))
                    st.write(top_k_from_probs(probs, class_names, k=top_k))
    else:
        st.warning("Missing folder `demo_images/`. Add sample images for this tab.")

# Labeled evaluation
elif mode == "Eval Folder (labeled)":
    st.info("Provide a folder where each subfolder is the label: e.g. eval_folder/Apple___healthy/*.jpg")
    eval_folder = st.text_input("Eval folder", value="eval_folder")
    run_eval = st.button("Run evaluation")
    if run_eval:
        y_true = []
        y_pred = []
        for label in os.listdir(eval_folder):
            lab_dir = os.path.join(eval_folder, label)
            if not os.path.isdir(lab_dir):
                continue
            for f in os.listdir(lab_dir):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    fp = os.path.join(lab_dir,f)
                    try:
                        img = Image.open(fp).convert("RGB")
                        probs = predict_np(model, np.array(img))
                        topk = top_k_from_probs(probs, class_names, k=1)
                        pred_label = topk[0][0] if topk else "UNKNOWN"
                    except:
                        pred_label = "ERROR"
                    y_true.append(label)
                    y_pred.append(pred_label)
        if len(y_true)==0:
            st.warning("No labeled images found.")
        else:
            st.subheader("Classification report")
            st.text(classification_report(y_true, y_pred, labels=list(set(y_true))))
            st.subheader("Confusion matrix")
            cmx = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Custom interface with image upload, batch prediction, Grad-CAM visualization, and evaluation tools.")

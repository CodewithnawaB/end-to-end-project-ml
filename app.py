"""
Streamlit App — Intel Image Classification (ML + ORB Feature Descriptors)
Loads the model trained in the Kaggle notebook and predicts the scene class
of an uploaded image.
"""

import os
import json
import numpy as np
import cv2
import joblib
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config — must match the notebook exactly
# ---------------------------------------------------------------------------
IMG_SIZE = 128
N_FEATURES = 500  # nfeatures used in cv2.ORB_create(nfeatures=500)
MODEL_PATH = "D:/end-to-end-projects/best_model.pkl"
CLASSES_PATH = "D:/end-to-end-projects/classes.json"

# Fallback class list (used only if classes.json is missing).
# NOTE: this MUST match the order produced by `os.listdir(data_dir)` in your
# notebook, since that order is what the model's labels (0,1,2,...) refer to.
# Safest fix: in your notebook, after `classes = os.listdir(data_dir)`, run:
#     import json
#     json.dump(classes, open('classes.json', 'w'))
# and place that classes.json next to this app.py file.
DEFAULT_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Place it next to app.py.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_CLASSES


def preprocess_image(img_bgr):
    """Same steps as the notebook's preprocess_image()."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def extract_orb_features(gray_img):
    """Same steps as the notebook's extract_orb_features()."""
    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    if descriptors is None:
        return np.zeros(N_FEATURES)
    descriptors = descriptors.flatten()
    if len(descriptors) < N_FEATURES:
        descriptors = np.pad(descriptors, (0, N_FEATURES - len(descriptors)))
    return descriptors[:N_FEATURES]


def get_confidence(model, features):
    """Return (predicted_label_idx, confidence_0_to_1)."""
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(features)[0]
            pred_idx = int(np.argmax(probs))
            return pred_idx, float(probs[pred_idx]), probs
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)[0]
        scores = np.atleast_1d(scores)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        pred_idx = int(np.argmax(probs))
        return pred_idx, float(probs[pred_idx]), probs
    pred_idx = int(model.predict(features)[0])
    return pred_idx, None, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Intel Image Classification", page_icon="🏔️", layout="centered")
st.title("🏔️ Intel Image Classification")
st.write(
    "Upload a photo and the model will classify it into one of 6 scene types: "
    "**buildings, forest, glacier, mountain, sea, street**."
)

model = load_model()
classes = load_classes()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("Extracting features and predicting..."):
        gray = preprocess_image(img_bgr)
        features = extract_orb_features(gray).reshape(1, -1)
        pred_idx, confidence, probs = get_confidence(model, features)

    pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

    st.success(f"### Prediction: **{pred_label.upper()}**")
    if confidence is not None:
        st.write(f"Confidence: **{confidence * 100:.1f}%**")
        if probs is not None and len(probs) == len(classes):
            st.bar_chart({cls: float(p) for cls, p in zip(classes, probs)})
    else:
        st.info(
            "This model wasn't trained with `probability=True`, so a numeric "
            "confidence score isn't available — only the predicted class."
        )

st.markdown("---")
st.caption("Model: SVM/RF/kNN trained on ORB feature vectors · Intel Image Classification dataset")

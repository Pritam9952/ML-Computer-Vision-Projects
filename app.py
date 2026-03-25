import streamlit as st
import cv2
import numpy as np
import pickle

from features import extract_features

# Page config
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")

# Custom styling (🔥 UI boost)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧬 Skin Cancer Classification App")
st.markdown("### 📊 AI-powered Skin Disease Detection System")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses Machine Learning to classify skin lesions into different categories.")

# Label mapping
label_map = {
    "nv": "Normal Mole (Benign)",
    "mel": "Melanoma (Cancer ⚠️)",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion"
}

# Load model
model = pickle.load(open("rf_model.pkl", "rb"))
if st.button("🔄 Reset App"):
    st.rerun()

# Upload
uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image
    st.image(img, caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Features
    features = extract_features([gray])

    # Prediction
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)
    confidence = np.max(proba) * 100

    result = label_map.get(prediction, prediction)

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    # Result display
    if prediction == "mel":
        st.error(f"⚠️ {result}")
    else:
        st.success(f"✅ {result}")

    # Confidence
    st.write(f"📊 Confidence: {confidence:.2f}%")

    if confidence < 60:
        st.warning("⚠️ Low confidence prediction. Result may not be reliable.")

    st.markdown("---")

    # Info box
    st.info("⚠️ This is an AI-based prediction. Please consult a medical professional for accurate diagnosis.")

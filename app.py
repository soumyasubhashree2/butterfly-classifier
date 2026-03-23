import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os
from PIL import Image

# ------------------ LOAD MODEL ------------------
model_path = os.path.join(os.getcwd(), "butterfly_species_model.keras")
model = tf.keras.models.load_model(model_path)

# ------------------ LOAD CLASS NAMES ------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ------------------ BUTTERFLY INFO ------------------
butterfly_info = {
    "MONARCH": {
        "name": "Monarch Butterfly",
        "desc": "Famous for long-distance migration across North America.",
        "habitat": "Meadows, fields, and open areas",
        "feature": "Orange wings with black veins"
    },
    "VICEROY": {
        "name": "Viceroy Butterfly",
        "desc": "Mimics monarch butterfly to avoid predators.",
        "habitat": "Wetlands and marshes",
        "feature": "Black line across hind wings"
    },
    "BLUE MORPHO": {
        "name": "Blue Morpho",
        "desc": "Known for its bright iridescent blue wings.",
        "habitat": "Tropical rainforests",
        "feature": "Large size with shimmering blue color"
    },
    "COMMON ROSE": {
        "name": "Common Rose",
        "desc": "A striking black butterfly with red body markings.",
        "habitat": "Gardens and forests in India",
        "feature": "Red body with black wings"
    },
    "CRIMSON ROSE": {
        "name": "Crimson Rose",
        "desc": "Black wings with crimson red spots.",
        "habitat": "South Asia",
        "feature": "Bright red markings"
    },
    "TIGER SWALLOWTAIL": {
        "name": "Tiger Swallowtail",
        "desc": "Large yellow butterfly with black tiger stripes.",
        "habitat": "Woodlands and gardens",
        "feature": "Tail-like extensions on wings"
    },
    "PAINTED LADY": {
        "name": "Painted Lady",
        "desc": "One of the most widespread butterflies in the world.",
        "habitat": "Open areas worldwide",
        "feature": "Orange and black pattern"
    },
    "RED ADMIRAL": {
        "name": "Red Admiral",
        "desc": "Fast-flying butterfly with red bands.",
        "habitat": "Gardens and forests",
        "feature": "Black wings with red stripes"
    },
    "GLASSWING": {
        "name": "Glasswing",
        "desc": "Transparent wings for camouflage.",
        "habitat": "Central and South America",
        "feature": "Clear wings"
    },
    "ZEBRA LONGWING": {
        "name": "Zebra Longwing",
        "desc": "Black and white striped butterfly.",
        "habitat": "Tropical regions",
        "feature": "Long wings with stripes"
    }
}

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Butterfly Classifier", page_icon="🦋", layout="wide")

st.markdown("<h1 style='text-align: center;'>🦋 Butterfly Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image or try a sample</p>", unsafe_allow_html=True)

# ------------------ SAMPLE BUTTON ------------------
use_sample = st.button("🖼 Try Sample Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

img = None

if use_sample:
    sample_path = "sample.jpg"
    if os.path.exists(sample_path):
        img = Image.open(sample_path)
        st.image(img, caption="Sample Image", use_column_width=True)
    else:
        st.error("Add sample.jpg to your project folder")

elif uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

# ------------------ PREDICTION ------------------
if img is not None:

    with st.spinner("Analyzing butterfly..."):
        img_resized = img.resize((224,224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        top_3 = prediction.argsort()[-3:][::-1]

    col1, col2 = st.columns(2)

    # -------- LEFT: PREDICTIONS --------
    with col1:
        st.subheader("🔍 Predictions")

        results = {}
        for i in top_3:
            results[class_names[i]] = float(prediction[i])

        st.bar_chart(results)

        top_class = class_names[top_3[0]]
        top_conf = prediction[top_3[0]] * 100

        st.success(f"🏆 Predicted: {top_class} ({top_conf:.2f}%)")

        # Confidence indicator
        if top_conf > 90:
            st.success("High Confidence")
        elif top_conf > 70:
            st.warning("Moderate Confidence")
        else:
            st.error("Low Confidence")

    # -------- RIGHT: INFO --------
    with col2:
        st.subheader("📚 Butterfly Information")

        info = butterfly_info.get(top_class.upper())

        if info:
            st.markdown(f"### 🦋 {info['name']}")
            st.write(f"**Description:** {info['desc']}")
            st.write(f"**Habitat:** {info['habitat']}")
            st.write(f"**Key Feature:** {info['feature']}")
        else:
            st.warning("No detailed information available for this species.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with TensorFlow + Streamlit</p>", unsafe_allow_html=True)


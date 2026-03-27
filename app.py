import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ✅ Load model (FIXED NAME)
model = tf.keras.models.load_model("butterfly_model.keras", compile=False)

# ✅ Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Butterfly info (optional)
butterfly_info = {
    "MONARCH": "A well-known orange and black butterfly famous for long migrations.",
    "VICEROY": "Looks similar to Monarch but has a black line across hind wings.",
    "MALACHITE": "A bright green butterfly commonly found in tropical regions."
}

def predict(image):
    if image is None:
        return "Please upload an image", ""

    try:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        top_3_idx = predictions.argsort()[-3:][::-1]

        results = []

        for i in top_3_idx:
            label = class_names[i]
            confidence = predictions[i] * 100
            results.append(f"{label} : {confidence:.2f}%")

        top_label = class_names[top_3_idx[0]]
        description = butterfly_info.get(top_label, "No description available.")

        return "\n".join(results), description

    except Exception as e:
        return f"Error: {str(e)}", ""

# ✅ Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=["text", "text"],
    title="🦋 Butterfly Species Classifier",
    description="Upload a butterfly image to predict species with confidence scores."
)

interface.launch()
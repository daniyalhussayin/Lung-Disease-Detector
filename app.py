import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# === Load Model ===
model = load_model("lung_disease_model.h5")

CATEGORIES = ["Normal", "Pneumonia"]
IMG_SIZE = 100

# === Prediction Function ===
def predict_image(image):
    img = np.array(image.convert("L"))  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    prediction = model.predict(img)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index] * 100

    return CATEGORIES[result_index], confidence


# === Streamlit UI ===
st.set_page_config(page_title="Lung Disease Detector", layout="centered")

st.title("🫁 Lung Disease Detector")
st.write("Upload a chest X-ray image to check for Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("---")

    if st.button("🔍 Predict"):
        result, confidence = predict_image(image)

        if result == "Normal":
            st.success(f"✅ Result: {result} ({confidence:.2f}%)")
        else:
            st.error(f"⚠️ Result: {result} ({confidence:.2f}%)")
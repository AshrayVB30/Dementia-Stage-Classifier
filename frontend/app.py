import streamlit as st
from Training import load_model, predict_image, get_suggestions
import os

st.title("🧠 Dementia Stage Classifier")
st.write("Upload a brain MRI image to predict the dementia stage.")

# Load model once
model = load_model()

if model is None:
    st.error("⚠️ Model could not be loaded. Please ensure 'dementia_model.h5' exists in the correct directory.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_image_path = "temp.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            # Make prediction
            label, confidence = predict_image(temp_image_path, model)
            behavior, precaution = get_suggestions(label)

            # Display results
            st.image("temp.jpg", caption="Uploaded MRI Image", use_container_width=True)
            st.markdown(f"### 🧾 Prediction: **{label}**")
            st.markdown(f"#### 🔢 Confidence: **{confidence * 100:.2f}%**")
            st.info(f"🧠 {behavior}")
            st.success(f"✅ {precaution}")
        except Exception as e:
            st.error(f"An error occurred while making a prediction: {e}")

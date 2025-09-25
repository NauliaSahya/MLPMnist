import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('mlp_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['W'], model_data['b'], model_data['alpha']

# Prediction function
def predict(W, b, alpha, img_data):
    a = [None] * (len(W) + 1)
    z = [None] * len(W)
    a[0] = img_data.reshape(-1, 1)

    for layer in range(len(W)):
        z[layer] = W[layer] @ a[layer] + b[layer]
        a[layer+1] = 1.0 / (1.0 + np.exp(-alpha * z[layer]))
    
    prediction = np.argmax(a[-1])
    return prediction, a[-1]

# Streamlit UI
st.set_page_config(page_title="MLP Digit Predictor", layout="centered")
st.title("MLP Digit Predictor")
st.markdown("Upload a handwritten digit image (0-9) to see the prediction.")

try:
    W, b, alpha = load_model()
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please run the training script first.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img_resized = img.resize((27, 28))
    
    st.subheader("Uploaded Image")
    st.image(img_resized, caption='Resized Image', use_column_width=False)
    
    img_array = np.array(img_resized).flatten()
    img_normalized = img_array / 255.0
    
    prediction, confidence_scores = predict(W, b, alpha, img_normalized)
    
    st.markdown("---")
    
    st.subheader("Prediction Result")
    st.markdown(f"The model predicts this is a: **<font color='blue' size='7'>{prediction}</font>**", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Confidence Scores")
    confidence_data = {
        'Digit': np.arange(10),
        'Confidence': np.squeeze(confidence_scores)
    }
    
    st.bar_chart(data=confidence_data, x='Digit', y='Confidence')
    st.write(f"Highest score: {round(np.max(confidence_scores) * 100, 2)}% for digit {prediction}.")
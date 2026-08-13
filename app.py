import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Diagnosis App", page_icon="🧬", layout="wide")
st.title("🧬 Nature-Inspired Feature Selection App")

# Load the saved .pkl model bundle
@st.cache_resource
def load_saved_model():
    with open('optimized_model.pkl', 'rb') as file:
        bundle = pickle.load(file)
    
    cancer = load_breast_cancer()
    X_full = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    
    return bundle['model'], bundle['scaler'], bundle['selected_indices'], X_full, cancer.feature_names

model, scaler, best_indices, X_full, feature_names = load_saved_model()

# Sidebar sliders for ONLY your optimized features
st.sidebar.header("Optimized Patient Measurements")
user_inputs = {}
for idx in best_indices:
    feature_name = feature_names[idx]
    user_inputs[idx] = st.sidebar.slider(
        feature_name, 
        float(X_full[feature_name].min()), 
        float(X_full[feature_name].max()), 
        float(X_full[feature_name].mean())
    )

st.subheader("📊 Live Prediction Result")

if st.button("Run Prediction", type="primary"):
    # Format input and scale it using the exact scaler from Colab
    raw_input_vector = np.array([[user_inputs[idx] for idx in best_indices]])
    
    # Predict using your Colab-trained model
    prediction = model.predict(raw_input_vector)
    probability = model.predict_proba(raw_input_vector)
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction[0] == 1:
            st.error("### Result: MALIGNANT")
        else:
            st.success("### Result: BENIGN")
    with col2:
        confidence = np.max(probability[0]) * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.info(f"Loaded from Colab Model (.pkl) using {len(best_indices)} features.")
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
    # 1. Collect user slider inputs into a dataframe or 2D array in the exact feature order
    raw_input_values = [user_inputs[idx] for idx in best_indices]
    
    # 2. Convert to a 2D array format for the scaler
    input_array = np.array([raw_input_values])
    
    # 3. CRITICAL: Scale the user input using the exact fitted scaler from Colab!
    scaled_input_array = scaler.transform(input_array)
    
    # 4. Predict using the scaled input vector
    prediction = model.predict(scaled_input_array)
    probability = model.predict_proba(scaled_input_array)
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction[0] == 1:
            st.error("### Result: MALIGNANT")
        else:
            st.success("### Result: BENIGN")
    with col2:
        confidence = np.max(probability[0]) * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.info(f"Model: KNN using {len(best_indices)} features (Scaled)")
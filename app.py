import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Breast Cancer Diagnosis App", page_icon="🧬", layout="wide")
st.title("🧬 Nature-Inspired Feature Selection App")

@st.cache_resource
def load_and_prepare():
    # 1. Load the raw dataset
    cancer = load_breast_cancer()
    X_full = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    
    # 2. Build a PERFECT fresh scaler directly from the raw data
    real_scaler = StandardScaler()
    real_scaler.fit(X_full)
    
    # 3. Load the model and indices from your Colab pickle file
    with open('optimized_model.pkl', 'rb') as file:
        bundle = pickle.load(file)
        
    return bundle['model'], bundle['selected_indices'], real_scaler, X_full, cancer.feature_names

# Unpack our working components
model, best_indices, scaler, X_full, feature_names = load_and_prepare()

st.sidebar.header("Optimized Patient Measurements")
user_inputs = {}

# Build sliders based ONLY on the features your optimizer selected
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
    # Collect the raw values from the user sliders
    raw_input_values = [user_inputs[idx] for idx in best_indices]
    
    # Start with a full row of mean values for all 30 original features
    full_row = X_full.mean().values.copy().reshape(1, -1)
    
    # Overwrite the specific optimized feature positions with the user's slider values
    for i, idx in enumerate(best_indices):
        full_row[0, idx] = raw_input_values[i]
        
    # Scale using the perfect scaler we built at the top of the script
    scaled_full_row = scaler.transform(full_row)
    
    # Extract ONLY the optimized feature columns from the scaled array
    scaled_input_array = scaled_full_row[:, best_indices]
    
    # Predict using your Colab-trained KNN model
    prediction = model.predict(scaled_input_array)
    probability = model.predict_proba(scaled_input_array)
    
    col1, col2 = st.columns(2)
    with col1:
        # BUG FIX: Target 0 is Malignant, Target 1 is Benign!
        if prediction[0] == 0:
            st.error("### Result: MALIGNANT")
        else:
            st.success("### Result: BENIGN")
            
    with col2:
        confidence = np.max(probability[0]) * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.info(f"Model: KNN using {len(best_indices)} features (Scaled)")

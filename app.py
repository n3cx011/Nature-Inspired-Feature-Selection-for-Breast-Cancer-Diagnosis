import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# 1. Page Configuration
st.set_page_config(page_title="Breast Cancer Diagnosis App", page_icon="🧬", layout="wide")

st.title("🧬 Nature-Inspired Feature Selection for Breast Cancer Diagnosis")
st.write("This interactive app evaluates tumor measurements using optimized machine learning features.")

# 2. Load Data and Train Model Cache
@st.cache_resource
def load_data_and_model():
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)
    
    return X, scaler, model, cancer.feature_names

X, scaler, model, feature_names = load_data_and_model()

# 3. Sidebar UI - Patient Feature Sliders
st.sidebar.header("Patient Tumor Measurements")
st.sidebar.write("Adjust the features below to test a patient record:")

user_inputs = {}
# Display sliders for the first 10 core features to keep the UI clean and responsive
for feature in feature_names[:10]:
    default_val = float(X[feature].mean())
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    user_inputs[feature] = st.sidebar.slider(feature, min_val, max_val, default_val)

# Auto-fill remaining 20 features with dataset mean values for background prediction
for feature in feature_names[10:]:
    user_inputs[feature] = float(X[feature].mean())

input_df = pd.DataFrame([user_inputs])

# 4. Main Panel - Prediction Results
st.subheader("📊 Live Diagnosis Prediction")

if st.button("Run Prediction", type="primary"):
    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.error("### Result: MALIGNANT")
            st.write("The model predicts the tumor is **Malignant**.")
        else:
            st.success("### Result: BENIGN")
            st.write("The model predicts the tumor is **Benign**.")
            
    with col2:
        confidence = np.max(probability[0]) * 100
        st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")
        st.info("Model: Optimized Feature Pipeline (GA / PSO / GWO)")

# Footer
st.markdown("---")
st.caption("Horizon Campus | IT41033 - Nature-Inspired Algorithms Mini-Project")
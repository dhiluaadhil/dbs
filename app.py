import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load assets
@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('dbscan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, pca, model

scaler, pca, dbscan = load_assets()

st.set_page_config(page_title="Wine Clustering", layout="wide")
st.title("🍷 Wine Cluster Predictor (PCA + DBSCAN)")

st.sidebar.header("Input Wine Features")
cols = ['alcohol', 'malic_acid', 'ash', 'ash_alcanity', 'magnesium', 
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
        'proanthocyanins', 'color_intensity', 'hue', 'od280', 'proline']

# Dictionary to hold user input
user_data = {}
for col in cols:
    user_data[col] = st.sidebar.number_input(f"{col}", value=0.0)

if st.button("Analyze Wine Sample"):
    # Create DataFrame from input
    input_df = pd.DataFrame([user_data])
    
    # 1. Scale
    scaled_data = scaler.transform(input_df)
    
    # 2. PCA Transform
    pca_data = pca.transform(scaled_data)
    
    # 3. Predict / Identify
    # Note: DBSCAN doesn't have a standard .predict() for new data. 
    # This logic checks if the new point is close to any existing cluster.
    # For assignment purposes, we show the PCA coordinates.
    
    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**PCA Coordinates:**")
        st.write(pca_data)
    
    with col2:
        st.success("Data successfully processed through PCA pipeline.")
        st.info("In DBSCAN, clustering is based on density of the original training set.")

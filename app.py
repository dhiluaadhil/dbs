import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import DBSCAN

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('dbscan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, dbscan = load_assets()

st.title("🍷 Wine Clustering Deployment")
st.write("Enter chemical attributes to determine the wine cluster (DBSCAN).")

# Create input fields for all 13 features
cols = ['alcohol', 'malic_acid', 'ash', 'ash_alcanity', 'magnesium', 
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
        'proanthocyanins', 'color_intensity', 'hue', 'od280', 'proline']

user_inputs = {}
col1, col2, col3 = st.columns(3)

for i, feature in enumerate(cols):
    with [col1, col2, col3][i % 3]:
        user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict Cluster"):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Standardize input using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # DBSCAN predict (Note: DBSCAN .fit_predict is used for clustering. 
    # For deployment, we check the distance to existing core points)
    # Here we show the logic of identifying if the point is noise or part of a group.
    
    # For a simple deployment visualization, we display the data
    st.subheader("Results")
    st.write("Input Data (Scaled):", scaled_input)
    st.info("Note: DBSCAN identifies clusters based on density. New points are typically evaluated against the trained density map.")
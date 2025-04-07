import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler  

import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
def load_model():
    try:
        model = joblib.load("kmeans_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# App title and description
st.title('Customer Segmentation using K-Means')
st.markdown("""This app predicts customer segments based on their characteristics.
Adjust the input parameters below and click **Predict Segment**.""")

# Sidebar for additional controls
with st.sidebar:
    st.header("Model Information")
    if model is not None:
        st.write(f"Number of clusters: {model.n_clusters}")
        st.write(f"Features expected: {model.n_features_in_}")
    else:
        st.warning("Model not loaded properly")
    
    st.header("Instructions")
    st.info("Fill all the fields with appropriate values for accurate prediction")

# Create input sections with better organization
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic Information")
    Age = st.number_input('Age', min_value=18, max_value=100, value=30)
    income = st.number_input('Income (USD)', min_value=0, value=50000)
    Education = st.selectbox('Education Level',[1, 2, 3, 4, 5],help="1: Basic, 2: High School, 3: Bachelor, 4: Master, 5: PhD")
    Total_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    Marital_Status_Together = st.selectbox('Marital Status', [0, 1], help="0: Single, 1: Together")
with col2:
    st.subheader("Behavioral Information")
    Recency = st.number_input('Days Since Last Purchase', min_value=0, value=30)
    Total_Spending = st.number_input('Total Spending (USD)', min_value=0.0, value=500.0)
    NumDealsPurchases = st.number_input('Number of Deal Purchases', min_value=0, value=2)
    NumWebPurchases = st.number_input('Number of Web Purchases', min_value=0, value=4)

st.subheader("Campaign Response")
col3, col4 = st.columns(2)
with col3:
    Membership_duration = st.number_input('Membership Duration (months)', min_value=0, value=12)
    Total_AcceptedCmp = st.number_input('Total Accepted Campaigns', min_value=0, max_value=10, value=1)
with col4:
    Complain = st.selectbox('Has Complained?', [0, 1], help="0: No, 1: Yes")
    Response = st.selectbox('Responded to Last Campaign?', [0, 1], help="0: No, 1: Yes")
    

# Prediction button with enhanced logic
if st.button("Predict Segment", type="primary"):
    if model is None:
        st.error("Model not available. Please check the model file path.")
    else:
        try:
            # Create feature array
            features = np.array([[Age, income, Education, Recency, Complain, Response, 
                Membership_duration, Total_AcceptedCmp, Total_Spending,
                Total_children, NumDealsPurchases, NumWebPurchases, 
                Marital_Status_Together]])
            
            # Check if model has the expected number of features
            if features.shape[1] != model.n_features_in_:
                st.error(f"Model expects {model.n_features_in_} features but got {features.shape[1]}")
            else:
                # Make prediction
                cluster = model.predict(features)
                
                # Display results with visualization
                st.success(f"**Predicted Customer Segment:** {cluster[0]}")
                
                # Add some interpretation
                segment_descriptions = {
                    0: "Budget-conscious, infrequent shoppers",
                    1: "High-value, loyal customers"}
                
                if cluster[0] in segment_descriptions:
                    st.info(f"**Segment Characteristics:** {segment_descriptions[cluster[0]]}")
                
                # Simple visualization
                fig, ax = plt.subplots()
                ax.barh(['Income', 'Total Spending'], 
                       [income, Total_Spending], 
                       color='skyblue')
                ax.set_title('Key Customer Metrics')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Prediction failed:{str(e)}")

# Add footer
st.markdown("---")

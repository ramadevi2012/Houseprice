import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model (Joblib)
model = joblib.load("house_price_model.joblib")

st.title("🏠 House Price Prediction App")
st.write("Enter the house area (in square feet) to predict the price")

# User input
area = st.number_input( "Enter area (sqft):",min_value=500,max_value=10000,step=100)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([[area]], columns=['area'])
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹ {np.round(prediction[0], 2)}")
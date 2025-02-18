import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and preprocessor
try:
    model = joblib.load(r"D:\ml_prjects\Projects\Ml_ops_Proj1\model\model.joblib")
    preprocessor = joblib.load(r"D:\ml_prjects\Projects\Ml_ops_Proj1\preprocessor.joblib")
    print("Model loaded successfully")
except:
    print("Model not found. Please train and save the model first.")
    exit()

st.title("Sales Prediction App")

# Input form for user
feature_inputs = {}
columns = preprocessor.transformers_[0][2]  # Get column names

for col in columns:
    feature_inputs[col] = st.number_input(f"{col}", value=0.0)

# Prediction button
if st.button("Predict"):
    input_data = pd.DataFrame([feature_inputs])
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]

    st.write(f"Predicted Units Sold: {prediction:.2f}")

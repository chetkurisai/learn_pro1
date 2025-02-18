import streamlit as st
import joblib
import pandas as pd
import numpy as np
# from google.cloud import aiplatform

# Initialize Vertex AI
# aiplatform.init(project="hip-wharf-447809-b7", location="us-central1") #Replace

# Load model and preprocessor (locally for demo, ideally from GCS)
try:
    model = joblib.load(r"D:\ml_prjects\Projects\Titanic_streamlit_deploy\model\model.joblib")
    preprocessor = joblib.load(r"D:\ml_prjects\Projects\Titanic_streamlit_deploy\preprocessor.joblib")
    print("Model loaded locally")
except:
    print("Model not found locally, please train and save model")
    exit()

st.title("Titanic Survival Prediction")

# Input form
pclass = st.selectbox("pclass", [1, 2, 3])
sex = st.selectbox("sex", ["male", "female"])
age = st.number_input("age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("sibsp", min_value=0, value=0)
parch = st.number_input("parch", min_value=0, value=0)
fare = st.number_input("fare", min_value=0.0, value=20.0)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Pclass': [pclass], 'Sex': [1 if sex == "female" else 0], 'Age': [age],
        'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare]
    })
    input_data['FamilySize'] = input_data['SibSp'] + input_data['Parch'] + 1

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]

    st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
pipeline = joblib.load('C:/Users/ugrkr/OneDrive/Masaüstü/model_pipeline.pkl')

import os
model_path = 'C:/Users/ugrkr/OneDrive/Masaüstü/model_pipeline.pkl'
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
else:
    print("Dosya bulunamadı:", model_path)
# Function to make predictions
def predict(input_data):
    prediction = pipeline.predict(input_data)
    return prediction

# Streamlit app title
st.title('Bank Churn Prediction App')

# Create input fields for the user
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=2)
estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=200000, value=2000)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
balance = st.number_input('Balance', min_value=-100000, max_value=100000, value=0)

# Convert 'Yes'/'No' to 1/0 for the model
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'EstimatedSalary': [estimated_salary],
    'NumOfProducts': [num_of_products],
    'CreditScore': [credit_score],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Balance': [balance]
})

# Make prediction
if st.button('Predict'):
    prediction = predict(input_data)
    st.write('Prediction:', 'Exited' if prediction[0] == 1 else 'Not Exited')

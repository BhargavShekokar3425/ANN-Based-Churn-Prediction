import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler
with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehotencoder_geography.pkl', 'rb') as f:
    onehotencoder_geography = pickle.load(f)

st.title('Customer Churn Prediction')

# User Input
gender = st.selectbox('Gender', ['Male', 'Female'])
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
age = st.number_input('Age', 18, 92)
tenure = st.number_input('Tenure', 0, 10)
balance = st.number_input('Balance')
NumOfProducts = st.number_input('Number of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [0],  # Replace with a dummy value or accept user input
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# One-hot encode 'Geography' and transform 'Gender'
geo_encoded = onehotencoder_geography.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'])

gender_encoded = label_encoder_gender.transform(input_data['Gender'])
input_data['Gender'] = gender_encoded

# Drop original 'Geography' column and concatenate encoded features
input_data = input_data.drop('Geography', axis=1)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Ensure columns match scaler's expectation
required_columns = scaler.feature_names_in_
missing_columns = [col for col in required_columns if col not in input_data.columns]

# Add missing columns with default values
for col in missing_columns:
    input_data[col] = 0

# Reorder columns to match the scaler's training order
input_data = input_data[required_columns]

# Preprocess the input data
input_data_scaled = scaler.transform(input_data)

# Make the prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')

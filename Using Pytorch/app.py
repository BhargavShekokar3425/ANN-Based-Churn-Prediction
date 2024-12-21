import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import torch
import torch.nn as nn

# Define the model architecture with input_size = 12
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.hidden1 = nn.Linear(12, 64)  # Input layer (12 input features) to first hidden layer (64 units)
        self.hidden2 = nn.Linear(64, 32)  # Second hidden layer (32 units)
        self.output = nn.Linear(32, 1)    # Output layer (1 unit for binary classification)
        self.sigmoid = nn.Sigmoid()       # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.hidden1(x))   # Apply ReLU activation on hidden1
        x = torch.relu(self.hidden2(x))   # Apply ReLU activation on hidden2
        x = self.output(x)                # Output layer
        x = self.sigmoid(x)               # Sigmoid for binary classification
        return x

# Instantiate the model
model = ChurnModel()

# Load the trained model's weights (state_dict)
model.load_state_dict(torch.load('model.pth'))  # Ensure this is the correct path to your model
model.eval()  # Set the model to evaluation mode


# Load the encoder and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('labelencoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehotencoder_geography.pkl', 'rb') as f:
    onehotencoder_geography = pickle.load(f)

# Define the Streamlit app
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

# Convert to tensor for PyTorch
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# Make the prediction with PyTorch model
with torch.no_grad():  # Disable gradient calculation since we are in inference mode
    prediction_prob = model(input_tensor).item()  # Get the probability as a scalar

if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is unlikely to churn.')

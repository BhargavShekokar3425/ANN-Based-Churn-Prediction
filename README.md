
# Crazy Cool ReadMe: Customer Churn Prediction App

Welcome to the **Customer Churn Prediction App**, a cutting-edge web application designed to predict whether a customer is likely to churn using the power of machine learning and deep learning! 🚀

---

## 🌟 Highlights
- **Interactive UI**: Built using Streamlit for a smooth user experience.
- **Deep Learning Model**: Powered by a trained TensorFlow model to make predictions with precision.
- **Custom Preprocessing**: Encoders and scalers ensure your inputs are just as the model likes them.
- **Real-World Dataset**: Developed using real-world churn data to deliver actionable insights.

---

## 🧩 Features

### 🔍 Inputs You Provide:
- **Gender**: Male or Female.
- **Geography**: Choose from France, Germany, or Spain.
- **Age**: A number between 18 and 92.
- **Tenure**: How long the customer has been with the company (0 to 10 years).
- **Balance**: The account balance of the customer.
- **Number of Products**: Number of products the customer uses (1 to 4).
- **Has Credit Card**: Whether the customer has a credit card (0 or 1).
- **Is Active Member**: Whether the customer is actively engaged (0 or 1).
- **Estimated Salary**: The estimated annual salary of the customer.

### 📊 Outputs You Get:
- **Prediction**: Whether the customer is likely to churn.
- **Probability**: How confident the model is in its prediction.

---

## 🛠️ Under the Hood

### 📂 File Overview:
- **`app.py`**: The main script that runs the Streamlit app.
- **`model.h5`**: Pre-trained TensorFlow model for churn prediction.
- **`standard_scaler.pkl`**: Scaler for standardizing numerical features.
- **`label_encoder_gender.pkl`**: Encoder to transform gender into numerical format.
- **`onehotencoder_geography.pkl`**: Encoder for geography feature.
- **`Churn_Modelling.csv`**: Original dataset used for training the model.
- **`experiments.ipynb`**: Jupyter notebook for training and experimentation.

### 🏗️ Data Flow:
1. **Input**: User enters data through Streamlit UI.
2. **Encoding**: Gender and Geography are encoded into numerical formats.
3. **Scaling**: Numerical features are standardized.
4. **Prediction**: Pre-processed input is fed into the model for prediction.
5. **Output**: Display results with churn probability.

### 📦 Dependencies:
- `streamlit`
- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`
- `pickle`

---

## 🚀 How to Run

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at the local address provided by Streamlit.

---

## 🎯 Use Case
This app is perfect for companies in subscription-based industries who want to:
- **Identify At-Risk Customers**: Take proactive steps to retain them.
- **Analyze Customer Data**: Get insights into customer behavior and trends.

---

## 🤔 FAQs

**1. What is customer churn?**
   - Customer churn occurs when a customer stops using a company's services. Predicting churn helps businesses take actions to retain customers.

**2. What dataset is used?**
   - The model is trained on the Kaggle Churn Modelling dataset.

**3. Can I add new features?**
   - Sure! Modify `app.py` and retrain the model as needed.

---

## 🛡️ Disclaimer
This app is intended for educational purposes only. Predictions are based on historical data and should not be used as a sole determinant for critical business decisions.

---

## 🎉 Have Fun!
Predict churn, analyze insights, and stay ahead of the curve. Cheers to smarter business decisions! 🥂


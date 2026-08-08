import streamlit as st
import joblib
import numpy as np

# Load the trained model and polynomial transformer
model = joblib.load("sales_model.pkl")
poly = joblib.load("poly_transform.pkl")

# Page title
st.title("Advertising Sales Predictor")

st.write("Enter your advertising budgets to predict product sales.")

# User inputs
tv = st.number_input("TV Advertising Budget", min_value=0.0, value=100.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0, value=20.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0, value=20.0)

# Prediction button
if st.button("Predict Sales"):
    new_budget = np.array([[tv, radio, newspaper]])

    # Transform input using polynomial features
    new_budget_poly = poly.transform(new_budget)

    # Predict sales
    prediction = model.predict(new_budget_poly)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")
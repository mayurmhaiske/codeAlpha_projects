import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    models = joblib.load("car_price_model.pkl")
    return models

model = load_model()

# -------------------------------
# App Title
# -------------------------------
st.title("Car Price Prediction App")
st.write("Enter the car details below to predict selling price")

# -------------------------------
# User Inputs
# -------------------------------
selling_price = st.number_input("Selling Price (in lakhs)", min_value=0.0, step=0.1)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
driven_kms = st.number_input("Driven KMs", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
selling_type = st.selectbox("Selling Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# -------------------------------
# Encoding (Must match training)
# -------------------------------
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
selling_map = {"Dealer": 0, "Individual": 1}
transmission_map = {"Manual": 0, "Automatic": 1}

fuel_type = fuel_map[fuel_type]
selling_type = selling_map[selling_type]
transmission = transmission_map[transmission]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    input_data = np.array([[selling_price,
                            present_price,
                            driven_kms,
                            fuel_type,
                            selling_type,
                            transmission,
                            owner]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")

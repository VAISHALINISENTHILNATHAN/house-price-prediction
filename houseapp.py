#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_price_xgb.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Predictor")

# Input fields
median_income = st.slider("Median Income (in $10,000s)", 0.0, 15.0, 5.0)
house_age = st.slider("House Age", 1, 52, 20)
avg_rooms = st.slider("Average Rooms", 1.0, 15.0, 6.0)
avg_bedrooms = st.slider("Average Bedrooms", 1.0, 5.0, 3.0)
population = st.slider("Population", 100, 10000, 1500)
avg_occupants = st.slider("Average Occupants", 1.0, 5.0, 3.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 35.0)
longitude = st.number_input("Longitude", -125.0, -114.0, -120.0)

if st.button("Predict Price"):
    # Derived features
    rooms_per_person = avg_rooms / (population + 1)
    bedrooms_per_room = avg_bedrooms / (avg_rooms + 1)
    population_density = population / (avg_occupants + 1)

    # Combine into one row (11 features)
    features = np.array([[median_income, house_age, avg_rooms,
                          avg_bedrooms, population, avg_occupants,
                          latitude, longitude,
                          rooms_per_person, bedrooms_per_room, population_density]])

    # Scale features before prediction
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    st.success(f"🏡 Predicted Median House Value: ${pred * 100000:.2f}")


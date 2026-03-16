import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and data
model = joblib.load("aqi_xgboost_model.pkl")
df = pd.read_csv("aqi_history.csv")

st.title("🌫️ AQI Prediction App")

# Dropdowns
states = df["State"].unique()
state = st.selectbox("Select State", states)

cities = df[df["State"] == state]["City"].unique()
city = st.selectbox("Select City", cities)

date = st.date_input("Select Date")
hour = st.slider("Hour", 0, 23, 12)

# Convert date
year = date.year
month = date.month
day = date.day
dayofweek = date.weekday()

# Get city encoding
city_enc = df[df["City"] == city]["city_enc"].iloc[0]
state_enc = df[df["City"] == city]["state_enc"].iloc[0]

# Get latest AQI values for city
city_data = df[df["City"] == city].sort_values(
    ["year","month","day","hour"]
)

AQI_lag_1 = city_data["AQI"].iloc[-1]
AQI_lag_24 = city_data["AQI"].iloc[-24]
AQI_roll_24 = city_data["AQI"].iloc[-24:].mean()

# Cyclical features
hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)

month_sin = np.sin(2*np.pi*month/12)
month_cos = np.cos(2*np.pi*month/12)

dow_sin = np.sin(2*np.pi*dayofweek/7)
dow_cos = np.cos(2*np.pi*dayofweek/7)

# Prediction
if st.button("Predict AQI"):

    features = [[
        state_enc,
        city_enc,
        year,
        month,
        day,
        hour,
        dayofweek,
        hour_sin,
        hour_cos,
        month_sin,
        month_cos,
        dow_sin,
        dow_cos,
        AQI_lag_1,
        AQI_lag_24,
        AQI_roll_24
    ]]

    prediction = model.predict(features)[0]

    st.success(f"Predicted AQI: {prediction:.2f}")

    # AQI category
    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 200:
        category = "Moderate"
    elif prediction <= 300:
        category = "Poor"
    elif prediction <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.write("AQI Category:", category)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------

API_KEY = "YOUR_AQICN_API_KEY"

# load model
model = joblib.load("aqi_xgb_model.pkl")

# -----------------------------
# LOAD HISTORY
# -----------------------------

if os.path.exists("aqi_history.csv"):
    df_hist = pd.read_csv("aqi_history.csv")
else:
    df_hist = pd.DataFrame(columns=["City","Datetime","AQI"])

# -----------------------------
# AQI API FUNCTION
# -----------------------------

def get_current_aqi(city):

    url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"

    try:
        r = requests.get(url)
        data = r.json()

        if data["status"] == "ok":
            return data["data"]["aqi"]
        else:
            return None

    except:
        return None

# -----------------------------
# UI
# -----------------------------

st.title("🌫 AI AQI Prediction")

st.write("Predict air quality using AI + real-time data")

cities = [
"delhi",
"ahmedabad",
"mumbai",
"bangalore",
"kolkata",
"chennai",
"hyderabad",
"pune"
]

city = st.selectbox("Select City", cities)

date = st.date_input("Select Date")

hour = st.slider("Hour",0,23,12)

year = date.year
month = date.month
day = date.day
dayofweek = date.weekday()

# -----------------------------
# FETCH REAL AQI
# -----------------------------

actual_aqi = get_current_aqi(city)

# -----------------------------
# UPDATE HISTORY
# -----------------------------

if actual_aqi is not None:

    new_row = {
        "City":city,
        "Datetime":datetime.now(),
        "AQI":actual_aqi
    }

    df_hist = pd.concat([df_hist,pd.DataFrame([new_row])])

    df_hist.to_csv("aqi_history.csv",index=False)

# -----------------------------
# BUILD LAG FEATURES
# -----------------------------

city_hist = df_hist[df_hist["City"]==city]

if len(city_hist) > 24:

    AQI_lag_1 = city_hist["AQI"].iloc[-1]
    AQI_lag_24 = city_hist["AQI"].iloc[-24]
    AQI_roll_24 = city_hist["AQI"].iloc[-24:].mean()

else:

    AQI_lag_1 = actual_aqi
    AQI_lag_24 = actual_aqi
    AQI_roll_24 = actual_aqi

# -----------------------------
# CYCLICAL FEATURES
# -----------------------------

hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)

month_sin = np.sin(2*np.pi*month/12)
month_cos = np.cos(2*np.pi*month/12)

dow_sin = np.sin(2*np.pi*dayofweek/7)
dow_cos = np.cos(2*np.pi*dayofweek/7)

# placeholder encodings (example)
state_enc = 0
city_enc = cities.index(city)

# -----------------------------
# PREDICT BUTTON
# -----------------------------

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

    st.subheader("Results")

    st.metric("Predicted AQI",round(prediction,2))
    st.metric("Current AQI",actual_aqi)

# -----------------------------
# AQI CATEGORY
# -----------------------------

    if prediction <= 50:
        cat = "Good"
    elif prediction <= 100:
        cat = "Satisfactory"
    elif prediction <= 200:
        cat = "Moderate"
    elif prediction <= 300:
        cat = "Poor"
    elif prediction <= 400:
        cat = "Very Poor"
    else:
        cat = "Severe"

    st.write("Category:",cat)

# -----------------------------
# SHOW HISTORY
# -----------------------------

st.subheader("Recent AQI Data")

st.dataframe(df_hist.tail(10))

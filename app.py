import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Air Quality Index Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

API_KEY = "06f7899efea26f9023918642e26799c5969ea9c6"

model = joblib.load("aqi_xgboost_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")

cities = sorted(list(city_encoder.classes_))

# ------------------------------------------------
# LOAD HISTORY
# ------------------------------------------------

if os.path.exists("aqi_history.csv"):
    df_hist = pd.read_csv("aqi_history.csv")
else:
    df_hist = pd.DataFrame(columns=["City","Datetime","AQI"])

# ------------------------------------------------
# AQI API FUNCTION
# ------------------------------------------------

def get_current_aqi(city):

    url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()

        if data.get("status") == "ok":
            return data["data"]["aqi"]

        return None

    except:
        return None


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("🌍 Air Quality Index Prediction Dashboard")
st.caption("Predict AQI using AI with real-time air pollution data")

st.divider()

# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("📍 City", cities)

with col2:
    date = st.date_input("📅 Date")

with col3:
    time_input = st.time_input("⏰ Time")
    hour = time_input.hour

year = date.year
month = date.month
day = date.day
dayofweek = date.weekday()

# ------------------------------------------------
# FETCH LIVE AQI
# ------------------------------------------------

city_api = city.lower().replace(" ", "-")

with st.spinner("Fetching live AQI..."):
    actual_aqi = get_current_aqi(city_api)

if actual_aqi is None:
    st.warning("⚠ Unable to fetch live AQI right now.")

# ------------------------------------------------
# UPDATE HISTORY
# ------------------------------------------------

if actual_aqi is not None:

    now = datetime.now()

    if len(df_hist) == 0 or df_hist.iloc[-1]["AQI"] != actual_aqi:

        new_row = {
            "City": city,
            "Datetime": now,
            "AQI": actual_aqi
        }

        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])])
        df_hist.to_csv("aqi_history.csv", index=False)

# ------------------------------------------------
# BUILD LAG FEATURES
# ------------------------------------------------

city_hist = df_hist[df_hist["City"] == city]

if len(city_hist) > 24:

    AQI_lag_1 = city_hist["AQI"].iloc[-1]
    AQI_lag_24 = city_hist["AQI"].iloc[-24]
    AQI_roll_24 = city_hist["AQI"].iloc[-24:].mean()

else:

    fallback = actual_aqi if actual_aqi is not None else 150

    AQI_lag_1 = fallback
    AQI_lag_24 = fallback
    AQI_roll_24 = fallback

# ------------------------------------------------
# CYCLICAL FEATURES
# ------------------------------------------------

hour_sin = np.sin(2*np.pi*hour/24)
hour_cos = np.cos(2*np.pi*hour/24)

month_sin = np.sin(2*np.pi*month/12)
month_cos = np.cos(2*np.pi*month/12)

dow_sin = np.sin(2*np.pi*dayofweek/7)
dow_cos = np.cos(2*np.pi*dayofweek/7)

# ------------------------------------------------
# ENCODING
# ------------------------------------------------

city_enc = city_encoder.transform([city])[0]
state_enc = 0

# ------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------

predict_btn = st.button("🚀 Predict AQI")

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if predict_btn:

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

    # AQI CATEGORY

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

    progress_value = float(min(prediction / 500, 1))

    col_left, col_right = st.columns([2,1])

    # RESULT PANEL

    with col_left:

        st.subheader("Predicted AQI")

        st.markdown(f"# {prediction:.2f}")

        st.markdown(f"### {category}")

        st.progress(progress_value)

        if actual_aqi is not None:
            st.metric("Current AQI (Live)", actual_aqi)

    # SUMMARY CARD

    with col_right:

        st.info(
f"""
**City:** {city}

**Date:** {date}

**Time:** {time_input}

**Predicted AQI:** {round(prediction,2)}

**Current AQI:** {actual_aqi}

**Category:** {category}
"""
        )

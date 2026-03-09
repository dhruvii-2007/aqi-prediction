import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==============================
# Page Setup
# ==============================

st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# ==============================
# Load Model Files
# ==============================

model = joblib.load("aqi_time_model.pkl")
features = joblib.load("aqi_features.pkl")
station_encoder = joblib.load("station_encoder.pkl")

# dataset for typical pollutant values
data = pd.read_csv("station_hour.csv")

# ==============================
# Header
# ==============================

st.title("🌍 Air Quality Index Prediction Dashboard")
st.markdown("Predict **Air Quality Index (AQI)** using station, time and pollutants")

st.divider()

# ==============================
# Input Section
# ==============================

col1, col2, col3 = st.columns(3)

station = col1.selectbox(
    "📍 Station",
    station_encoder.classes_
)

date = col2.date_input("📅 Date")

time = col3.time_input("⏰ Time")

# ==============================
# Optional Pollutants
# ==============================

with st.expander("⚙️ Advanced Pollutant Inputs (Optional)"):

    r1c1,r1c2,r1c3 = st.columns(3)

    pm25 = r1c1.number_input("PM2.5",0.0,1000.0,0.0)
    pm10 = r1c2.number_input("PM10",0.0,1000.0,0.0)
    no2 = r1c3.number_input("NO2",0.0,500.0,0.0)

    r2c1,r2c2,r2c3 = st.columns(3)

    so2 = r2c1.number_input("SO2",0.0,500.0,0.0)
    o3 = r2c2.number_input("O3",0.0,500.0,0.0)
    nox = r2c3.number_input("NOx",0.0,500.0,0.0)

predict = st.button("🚀 Predict AQI")

# ==============================
# Feature Engineering
# ==============================

def create_features():

    dt = datetime.combine(date,time)

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour

    dayofweek = dt.weekday()
    week = dt.isocalendar()[1]
    quarter = (month-1)//3 + 1
    dayofyear = dt.timetuple().tm_yday

    weekend = 1 if dayofweek >=5 else 0

    hour_sin = np.sin(2*np.pi*hour/24)
    hour_cos = np.cos(2*np.pi*hour/24)

    month_sin = np.sin(2*np.pi*month/12)
    month_cos = np.cos(2*np.pi*month/12)

    day_sin = np.sin(2*np.pi*dayofyear/365)
    day_cos = np.cos(2*np.pi*dayofyear/365)

    station_encoded = station_encoder.transform([station])[0]

    is_peak = 1 if hour in [7,8,9,17,18,19] else 0
    time_block = 0 if hour<6 else 1 if hour<12 else 2 if hour<18 else 3

    # ==============================
    # Typical pollutant values
    # ==============================

    station_data = data[data["StationId"] == station]

    pm25_val = pm25 if pm25>0 else station_data["PM2.5"].median()
    pm10_val = pm10 if pm10>0 else station_data["PM10"].median()
    no2_val = no2 if no2>0 else station_data["NO2"].median()
    so2_val = so2 if so2>0 else station_data["SO2"].median()
    o3_val = o3 if o3>0 else station_data["O3"].median()
    nox_val = nox if nox>0 else station_data["NOx"].median()

    aqi_typical = station_data["AQI"].median()

    pm25_pm10_ratio = pm25_val/(pm10_val+0.01)
    nox_no2_ratio = nox_val/(no2_val+0.01)

    data_dict = {

        'Station_encoded':station_encoded,

        'year':year,
        'month':month,
        'day':day,
        'hour':hour,

        'dayofweek':dayofweek,
        'week':week,
        'quarter':quarter,
        'dayofyear':dayofyear,

        'weekend':weekend,

        'hour_sin':hour_sin,
        'hour_cos':hour_cos,

        'month_sin':month_sin,
        'month_cos':month_cos,

        'day_sin':day_sin,
        'day_cos':day_cos,

        'is_peak':is_peak,
        'time_block':time_block,

        'pm25_pm10_ratio':pm25_pm10_ratio,
        'nox_no2_ratio':nox_no2_ratio,

        'PM2.5_typical':pm25_val,
        'PM10_typical':pm10_val,
        'NO2_typical':no2_val,
        'SO2_typical':so2_val,
        'O3_typical':o3_val,

        'aqi_rolling_6h':aqi_typical,
        'aqi_rolling_24h':aqi_typical,
        'aqi_rolling_168h':aqi_typical,

        'AQI_typical':aqi_typical
    }

    df = pd.DataFrame([data_dict])

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

# ==============================
# AQI Category
# ==============================

def get_category(aqi):

    if aqi<=50:
        return "Good","🟢"
    elif aqi<=100:
        return "Satisfactory","🟡"
    elif aqi<=200:
        return "Moderate","🟠"
    elif aqi<=300:
        return "Poor","🔴"
    elif aqi<=400:
        return "Very Poor","🟣"
    else:
        return "Severe","⚫"

# ==============================
# Prediction
# ==============================

if predict:

    X = create_features()

    pred_log = model.predict(X)[0]

    aqi = float(np.expm1(pred_log))

    category,emoji = get_category(aqi)

    st.divider()

    col1,col2 = st.columns([2,1])

    with col1:

        st.metric("Predicted AQI",f"{aqi:.2f}")
        st.subheader(f"{emoji} {category}")
        st.progress(min(aqi/500,1.0))

    with col2:

        st.info(f"""
Station: {station}

Date: {date}

Time: {time}

Category: **{category}**
""")

st.divider()

st.caption("Machine Learning AQI Prediction System • Built with Streamlit")

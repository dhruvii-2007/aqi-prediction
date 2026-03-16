import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------------------------
# PROFESSIONAL UI STYLE
# ------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Card UI */
.card{
    background:#111;
    padding:20px;
    border-radius:14px;
    border:1px solid #2a2a2a;
    box-shadow:0px 4px 12px rgba(0,0,0,0.4);
}

/* Inputs */
div[data-baseweb="select"]{
    border-radius:10px;
}

.stDateInput input{
    border-radius:10px;
}

.stTimeInput input{
    border-radius:10px;
}

/* Button styling */
.stButton>button{
    width:100%;
    height:48px;
    border-radius:10px;
    font-weight:600;
    font-size:16px;
}

/* AQI number */
.big-font{
    font-size:90px;
    font-weight:700;
}

/* Summary box */
.summary-box{
    background:#111;
    padding:20px;
    border-radius:12px;
    border:1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

API_KEY = "06f7899efea26f9023918642e26799c5969ea9c6"

model = joblib.load("aqi_xgboost_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")

cities = sorted(list(city_encoder.classes_))

# ------------------------------------------------
# AQI API
# ------------------------------------------------

def get_current_aqi(city):

    url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"

    try:
        r = requests.get(url, timeout=8)
        data = r.json()

        if data.get("status") == "ok":
            return data["data"]["aqi"]

        return None

    except:
        return None


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("🌍 Air Quality Prediction Dashboard")
st.caption("AI Powered Air Quality Forecasting")

st.divider()

# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------

st.subheader("Prediction Inputs")

container = st.container()

with container:

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox(
            "City",
            cities,
            help="Choose city for AQI prediction"
        )

    with col2:
        date = st.date_input(
            "Date",
            datetime.today()
        )

    with col3:
        time = st.time_input(
            "Time",
            datetime.now().time()
        )
        hour = time.hour

st.write("")

predict = st.button("🚀 Predict AQI")

# ------------------------------------------------
# DATE FEATURES
# ------------------------------------------------

year = date.year
month = date.month
day = date.day
dayofweek = date.weekday()

# ------------------------------------------------
# ENCODING
# ------------------------------------------------

city_enc = city_encoder.transform([city])[0]
state_enc = 0

# ------------------------------------------------
# AQI CATEGORY FUNCTION
# ------------------------------------------------

def aqi_category(aqi):

    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Satisfactory", "🟡"
    elif aqi <= 200:
        return "Moderate", "🟠"
    elif aqi <= 300:
        return "Poor", "🔴"
    elif aqi <= 400:
        return "Very Poor", "🟣"
    else:
        return "Severe", "⚫"

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if predict:

    city_api = city.lower().replace(" ", "-")

    with st.spinner("Fetching live AQI..."):
        actual_aqi = get_current_aqi(city_api)

    if actual_aqi is None:
        actual_aqi = 150
        st.warning("Live AQI unavailable. Using fallback value.")

    AQI_lag_1 = actual_aqi
    AQI_lag_24 = actual_aqi
    AQI_roll_24 = actual_aqi

    hour_sin = np.sin(2*np.pi*hour/24)
    hour_cos = np.cos(2*np.pi*hour/24)

    month_sin = np.sin(2*np.pi*month/12)
    month_cos = np.cos(2*np.pi*month/12)

    dow_sin = np.sin(2*np.pi*dayofweek/7)
    dow_cos = np.cos(2*np.pi*dayofweek/7)

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

    category, emoji = aqi_category(prediction)

    progress_value = int(min(prediction / 500 * 100, 100))

    # ------------------------------------------------
    # RESULTS UI
    # ------------------------------------------------

    colA, colB = st.columns([2,1])

    with colA:

        st.markdown("### Predicted AQI")

        st.markdown(
            f"<div class='big-font'>{round(prediction)}</div>",
            unsafe_allow_html=True
        )

        st.markdown(f"### {emoji} {category}")

        st.progress(progress_value)

    with colB:

        st.markdown("### Prediction Summary")

        st.markdown(
            f"""
<div class="summary-box">

City: **{city}**

Date: **{date}**

Time: **{time}**

Current AQI: **{actual_aqi}**

Predicted AQI: **{round(prediction)}**

Category: **{category}**

</div>
""",
            unsafe_allow_html=True
        )


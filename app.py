import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AQI Predictor - India",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    """Load model and artifacts from root directory"""
    try:
        model = joblib.load('aqi_model.pkl')
        feature_cols = joblib.load('aqi_features.pkl')
        station_encoder = joblib.load('station_encoder.pkl')
        return model, feature_cols, station_encoder
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load model
model, feature_cols, station_encoder = load_model()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_aqi(year, month, day, hour, minute, station_id=None):
    """Predict AQI using date components"""
    if model is None:
        return None
    
    # Create datetime
    dt = datetime(year, month, day, hour, minute)
    
    # Create feature dictionary
    features = {}
    
    # Basic time features
    features['year'] = dt.year
    features['month'] = dt.month
    features['day'] = dt.day
    features['hour'] = dt.hour
    features['dayofweek'] = dt.weekday()
    features['week'] = dt.isocalendar().week
    features['quarter'] = (dt.month - 1) // 3 + 1
    features['dayofyear'] = dt.timetuple().tm_yday
    features['weekend'] = 1 if dt.weekday() >= 5 else 0
    
    # Cyclical features
    features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
    features['day_sin'] = np.sin(2 * np.pi * features['dayofyear'] / 365)
    features['day_cos'] = np.cos(2 * np.pi * features['dayofyear'] / 365)
    
    # Time context
    features['is_peak'] = 1 if dt.hour in [7, 8, 9, 17, 18, 19] else 0
    features['time_block'] = 0 if dt.hour <= 5 else (1 if dt.hour <= 11 else (2 if dt.hour <= 17 else 3))
    
    # Station encoding
    if station_id and station_id in station_encoder.classes_:
        features['Station_encoded'] = station_encoder.transform([station_id])[0]
    else:
        features['Station_encoded'] = len(station_encoder.classes_) // 2
    
    # Default values for engineered features
    default_features = {
        'aqi_rolling_6h': 150,
        'aqi_rolling_24h': 150,
        'aqi_rolling_168h': 150,
        'PM2.5_typical': 60,
        'PM10_typical': 120,
        'NO2_typical': 30,
        'SO2_typical': 10,
        'O3_typical': 35,
        'AQI_typical': 150,
        'pm25_pm10_ratio': 0.5,
        'nox_no2_ratio': 1.5
    }
    
    for col, val in default_features.items():
        if col in feature_cols:
            features[col] = val
    
    # Create dataframe
    input_data = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], 
                              columns=feature_cols)
    
    # Predict
    pred_log = model.predict(input_data)[0]
    aqi = np.expm1(pred_log)
    
    return aqi

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "green", "Air quality is satisfactory. No health risks."
    elif aqi <= 100:
        return "Satisfactory", "lightgreen", "Minor discomfort for sensitive individuals."
    elif aqi <= 200:
        return "Moderate", "gold", "Sensitive individuals should reduce outdoor activities."
    elif aqi <= 300:
        return "Poor", "orange", "Everyone may experience health effects. Sensitive groups should avoid outdoor activities."
    elif aqi <= 400:
        return "Very Poor", "red", "Health warnings. Everyone may experience serious health effects."
    else:
        return "Severe", "darkred", "Emergency conditions. Everyone should avoid outdoor activities."

# ============================================
# UI HEADER
# ============================================
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🌍 AQI Predictor - India</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #0D47A1; font-size: 1.2rem;'>Predict Air Quality Index for any date, time, and location</p>", unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUTS (Using text inputs instead of date_input)
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/air-quality.png", width=100)
    st.title("📝 Input Parameters")
    
    # Date inputs - using number inputs instead of date picker
    st.subheader("📅 Date")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Year", min_value=2015, max_value=2025, value=2024)
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
    
    # Time inputs
    st.subheader("⏰ Time")
    col1, col2 = st.columns(2)
    with col1:
        hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
    with col2:
        minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
    
    # Station selection
    if model is not None and station_encoder is not None:
        stations = list(station_encoder.classes_)
        station_input = st.selectbox("🏭 Select Station", stations, index=0)
    else:
        station_input = st.text_input("🏭 Enter Station ID", "Station_101")
    
    # Predict button
    predict_btn = st.button("🔮 Predict AQI", use_container_width=True)

# ============================================
# MAIN CONTENT
# ============================================
if model is None:
    st.error("❌ Model files not found!")
    st.info("Please ensure model files are in the correct location.")
    st.stop()

if predict_btn:
    with st.spinner("🔄 Predicting AQI..."):
        try:
            # Validate date
            try:
                # Quick validation
                datetime(year, month, day, hour, minute)
            except ValueError as e:
                st.error(f"❌ Invalid date/time: {str(e)}")
                st.stop()
            
            # Make prediction
            aqi = predict_aqi(int(year), int(month), int(day), int(hour), int(minute), station_input)
            
            if aqi is None:
                st.error("Prediction failed")
                st.stop()
                
            category, color, advice = get_aqi_category(aqi)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("AQI Value", f"{aqi:.1f}")
            
            with col2:
                st.markdown(f"<h3 style='text-align: center; color: {color};'>{category}</h3>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"<h3 style='text-align: center;'>{station_input}</h3>", unsafe_allow_html=True)
                st.caption(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}")
            
            # Health advice
            st.info(f"💡 **Health Advice:** {advice}")
            
            # AQI Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "AQI Level"},
                gauge={
                    'axis': {'range': [0, 500]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "green"},
                        {'range': [50, 100], 'color': "lightgreen"},
                        {'range': [100, 200], 'color': "gold"},
                        {'range': [200, 300], 'color': "orange"},
                        {'range': [300, 400], 'color': "red"},
                        {'range': [400, 500], 'color': "darkred"}
                    ]
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
else:
    st.info("👈 Enter parameters in the sidebar and click 'Predict AQI'")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | Model Accuracy: R² = 0.974</p>",
    unsafe_allow_html=True
)
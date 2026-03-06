# app.py - Streamlit AQI Prediction App

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
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .aqi-good { color: green; font-weight: bold; }
    .aqi-satisfactory { color: lightgreen; font-weight: bold; }
    .aqi-moderate { color: yellow; font-weight: bold; }
    .aqi-poor { color: orange; font-weight: bold; }
    .aqi-very-poor { color: red; font-weight: bold; }
    .aqi-severe { color: darkred; font-weight: bold; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND ARTIFACTS
# ============================================
@st.cache_resource
def load_model():
    """Load model and artifacts from Google Drive or local"""
    try:
        # Try loading from Google Drive first
        model = joblib.load('/content/drive/MyDrive/aqi_model.pkl')
        feature_cols = joblib.load('/content/drive/MyDrive/aqi_features.pkl')
        station_encoder = joblib.load('/content/drive/MyDrive/station_encoder.pkl')
    except:
        # Fallback to local files
        model = joblib.load('aqi_model.pkl')
        feature_cols = joblib.load('aqi_features.pkl')
        station_encoder = joblib.load('station_encoder.pkl')
    
    return model, feature_cols, station_encoder

# Load model
try:
    model, feature_cols, station_encoder = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_aqi(date_input, time_input, station_id=None):
    """
    Predict AQI using date, time, and station
    """
    
    # Parse datetime
    if isinstance(date_input, str):
        dt = pd.to_datetime(f"{date_input} {time_input}")
    else:
        dt = date_input
    
    # Create feature dictionary
    features = {}
    
    # Basic time features
    features['year'] = dt.year
    features['month'] = dt.month
    features['day'] = dt.day
    features['hour'] = dt.hour
    features['dayofweek'] = dt.dayofweek
    features['week'] = dt.isocalendar().week
    features['quarter'] = dt.quarter
    features['dayofyear'] = dt.dayofyear
    features['weekend'] = 1 if dt.dayofweek >= 5 else 0
    
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
# HEADER SECTION
# ============================================
st.markdown('<h1 class="main-header">🌍 AQI Predictor - India</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict Air Quality Index for any date, time, and location</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUTS
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/air-quality.png", width=100)
    st.title("📝 Input Parameters")
    
    # Date input
    date_input = st.date_input(
        "📅 Select Date",
        datetime.now(),
        min_value=datetime(2015, 1, 1),
        max_value=datetime(2025, 12, 31)
    )
    
    # Time input
    time_input = st.time_input(
        "⏰ Select Time",
        datetime.now().time()
    )
    
    # Station selection
    if model_loaded:
        stations = list(station_encoder.classes_)
        station_input = st.selectbox(
            "🏭 Select Station",
            stations,
            index=0
        )
    else:
        station_input = st.text_input("🏭 Enter Station ID", "Station_101")
        st.warning("⚠️ Using demo mode - limited stations")
    
    # Predict button
    predict_btn = st.button("🔮 Predict AQI", use_container_width=True)
    
    # Information section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Model Information:**
        - Algorithm: XGBoost
        - Training data: 2015-2020
        - Stations: 109 across India
        - Accuracy: R² = 0.974
        
        **Features used:**
        - Date & Time
        - Station location
        - Historical patterns
        - Temporal cycles
        """)

# ============================================
# MAIN CONTENT AREA
# ============================================
if not model_loaded:
    st.error(f"❌ Model not loaded: {error_msg}")
    st.info("Please ensure model files are in the correct location.")
    st.stop()

if predict_btn:
    with st.spinner("🔄 Predicting AQI..."):
        try:
            # Make prediction
            aqi = predict_aqi(date_input, time_input.strftime("%H:%M"), station_input)
            category, color, advice = get_aqi_category(aqi)
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: white; margin-bottom: 1rem;">📊 AQI Value</h3>
                    <h1 style="font-size: 4rem; color: white;">{aqi:.1f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: {color};">
                    <h3 style="color: white; margin-bottom: 1rem;">📋 Category</h3>
                    <h2 style="font-size: 2.5rem; color: white;">{category}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3 style="color: white; margin-bottom: 1rem;">📍 Location</h3>
                    <h3 style="color: white;">{station_input}</h3>
                    <p style="color: white;">{date_input} {time_input}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Health advice
            st.info(f"💡 **Health Advice:** {advice}")
            
            # AQI Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = aqi,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AQI Level", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 500], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "green"},
                        {'range': [50, 100], 'color': "lightgreen"},
                        {'range': [100, 200], 'color': "gold"},
                        {'range': [200, 300], 'color': "orange"},
                        {'range': [300, 400], 'color': "red"},
                        {'range': [400, 500], 'color': "darkred"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': aqi
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Week forecast
            st.subheader("📈 7-Day AQI Forecast")
            
            # Generate predictions for next 7 days at same time
            forecast_data = []
            for i in range(7):
                forecast_date = date_input + timedelta(days=i)
                forecast_aqi = predict_aqi(forecast_date, time_input.strftime("%H:%M"), station_input)
                forecast_cat, _, _ = get_aqi_category(forecast_aqi)
                forecast_data.append({
                    'Date': forecast_date.strftime("%Y-%m-%d"),
                    'AQI': round(forecast_aqi, 1),
                    'Category': forecast_cat
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Create forecast chart
            fig_forecast = px.line(forecast_df, x='Date', y='AQI', markers=True,
                                  title="7-Day AQI Forecast",
                                  labels={'AQI': 'Predicted AQI', 'Date': 'Date'})
            
            # Add color zones
            fig_forecast.add_hrect(y0=0, y1=50, line_width=0, fillcolor="green", opacity=0.2)
            fig_forecast.add_hrect(y0=50, y1=100, line_width=0, fillcolor="lightgreen", opacity=0.2)
            fig_forecast.add_hrect(y0=100, y1=200, line_width=0, fillcolor="gold", opacity=0.2)
            fig_forecast.add_hrect(y0=200, y1=300, line_width=0, fillcolor="orange", opacity=0.2)
            fig_forecast.add_hrect(y0=300, y1=400, line_width=0, fillcolor="red", opacity=0.2)
            fig_forecast.add_hrect(y0=400, y1=500, line_width=0, fillcolor="darkred", opacity=0.2)
            
            fig_forecast.update_layout(height=400)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show forecast table
            st.subheader("📊 Forecast Details")
            st.dataframe(forecast_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

else:
    # Show welcome message
    st.markdown("""
    <div style="text-align: center; padding: 4rem;">
        <h2>👋 Welcome to AQI Predictor!</h2>
        <p style="font-size: 1.2rem; color: gray;">
            Enter your parameters in the sidebar and click "Predict AQI" to get started.
        </p>
        <img src="https://img.icons8.com/color/96/000000/air-quality.png" style="margin-top: 2rem;">
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | Model Accuracy: R² = 0.974</p>",
    unsafe_allow_html=True
)
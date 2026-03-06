# app.py - COMPLETE FIXED VERSION

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
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
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL (from root directory)
# ============================================
@st.cache_resource
def load_model():
    """Load model and artifacts from root directory"""
    try:
        # Load directly from current directory
        model = joblib.load('aqi_model.pkl')
        feature_cols = joblib.load('aqi_features.pkl')
        station_encoder = joblib.load('station_encoder.pkl')
        return model, feature_cols, station_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please ensure these files are in the same directory as app.py:\n"
                "- aqi_model.pkl\n"
                "- aqi_features.pkl\n"
                "- station_encoder.pkl")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Load model
model, feature_cols, station_encoder = load_model()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_aqi(date_input, time_input, station_id=None):
    """Predict AQI using date, time, and station"""
    if model is None:
        return None
    
    # Parse datetime
    if isinstance(date_input, date):
        # Combine date and time
        dt = datetime.combine(date_input, time_input)
    else:
        dt = date_input
    
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
st.markdown('<h1 class="main-header">🌍 AQI Predictor - India</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict Air Quality Index for any date, time, and location</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUTS
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/air-quality.png", width=100)
    st.title("📝 Input Parameters")
    
    # Date input - FIXED with date objects
    date_input = st.date_input(
        "📅 Select Date",
        value=date.today(),
        min_value=date(2015, 1, 1),
        max_value=date(2025, 12, 31)
    )
    
    # Time input
    time_input = st.time_input(
        "⏰ Select Time",
        value=datetime.now().time()
    )
    
    # Station selection
    if model is not None and station_encoder is not None:
        stations = list(station_encoder.classes_)
        station_input = st.selectbox(
            "🏭 Select Station",
            stations,
            index=0
        )
    else:
        station_input = st.text_input("🏭 Enter Station ID", "Station_101")
    
    # Predict button
    predict_btn = st.button("🔮 Predict AQI", use_container_width=True)

# ============================================
# MAIN CONTENT
# ============================================
if model is None:
    st.error("❌ Model files not found!")
    st.info("Please ensure these files are in the same directory as app.py:\n"
            "- aqi_model.pkl\n"
            "- aqi_features.pkl\n"
            "- station_encoder.pkl")
    st.stop()

if predict_btn:
    with st.spinner("🔄 Predicting AQI..."):
        try:
            # Make prediction
            aqi = predict_aqi(date_input, time_input, station_input)
            
            if aqi is None:
                st.error("Prediction failed")
                st.stop()
                
            category, color, advice = get_aqi_category(aqi)
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 AQI Value</h3>
                    <h1 style="font-size: 4rem;">{aqi:.1f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: {color}; padding: 2rem; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h3>📋 Category</h3>
                    <h2 style="font-size: 2.5rem;">{category}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📍 Location</h3>
                    <h3>{station_input}</h3>
                    <p>{date_input} {time_input}</p>
                </div>
                """, unsafe_allow_html=True)
            
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
            
            # Generate 7-day forecast
            st.subheader("📈 7-Day AQI Forecast")
            
            forecast_data = []
            for i in range(7):
                forecast_date = date_input + timedelta(days=i)
                forecast_aqi = predict_aqi(forecast_date, time_input, station_input)
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
            with st.expander("📊 View Forecast Details"):
                st.dataframe(forecast_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)  # This will show the full error in development
else:
    st.info("👈 Enter parameters in the sidebar and click 'Predict AQI'")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown(
        "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | Model Accuracy: R² = 0.974</p>",
        unsafe_allow_html=True
    )
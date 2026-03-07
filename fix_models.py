import joblib
import numpy as np
import xgboost as xgb

print(f"NumPy version: {np.__version__}")

print("Loading model...")
model = joblib.load('aqi_model.pkl')
print("Loading features...")
features = joblib.load('aqi_features.pkl')
print("Loading encoder...")
encoder = joblib.load('station_encoder.pkl')

print("Saving fixed files...")
joblib.dump(model, 'aqi_model_fixed.pkl', compress=3)
joblib.dump(features, 'aqi_features_fixed.pkl', compress=3)
joblib.dump(encoder, 'station_encoder_fixed.pkl', compress=3)

print("✅ Files resaved with numpy 1.23.5")

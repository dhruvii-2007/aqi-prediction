import joblib
import numpy as np
import os

print(f"NumPy version: {np.__version__}")
print("Files before:", os.listdir('.'))

# Load and resave each file
print("\nLoading model...")
model = joblib.load('aqi_model.pkl')
print("✅ Model loaded")

print("Loading features...")
features = joblib.load('aqi_features.pkl')
print("✅ Features loaded")

print("Loading encoder...")
encoder = joblib.load('station_encoder.pkl')
print("✅ Encoder loaded")

# Save with current numpy version
print("\nResaving files...")
joblib.dump(model, 'aqi_model_new.pkl', compress=3)
joblib.dump(features, 'aqi_features_new.pkl', compress=3)
joblib.dump(encoder, 'station_encoder_new.pkl', compress=3)

# Replace old files
os.replace('aqi_model_new.pkl', 'aqi_model.pkl')
os.replace('aqi_features_new.pkl', 'aqi_features.pkl')
os.replace('station_encoder_new.pkl', 'station_encoder.pkl')

print("\n✅ Files fixed!")
print("Files after:", os.listdir('.'))

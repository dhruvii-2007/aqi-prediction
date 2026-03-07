@echo off
echo Installing required packages...
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 xgboost==1.7.6 joblib==1.2.0

echo Creating fix script...
echo import joblib > fix_models.py
echo import numpy as np >> fix_models.py
echo import xgboost as xgb >> fix_models.py
echo. >> fix_models.py
echo print(f"NumPy version: {np.__version__}") >> fix_models.py
echo. >> fix_models.py
echo print("Loading original models...") >> fix_models.py
echo model = joblib.load('aqi_model.pkl') >> fix_models.py
echo features = joblib.load('aqi_features.pkl') >> fix_models.py
echo encoder = joblib.load('station_encoder.pkl') >> fix_models.py
echo. >> fix_models.py
echo print("Saving fixed versions...") >> fix_models.py
echo joblib.dump(model, 'aqi_model_fixed.pkl', compress=3) >> fix_models.py
echo joblib.dump(features, 'aqi_features_fixed.pkl', compress=3) >> fix_models.py
echo joblib.dump(encoder, 'station_encoder_fixed.pkl', compress=3) >> fix_models.py
echo. >> fix_models.py
echo print("✅ Files resaved with numpy 1.23.5") >> fix_models.py

echo Running fix script...
python fix_models.py

echo Replacing old files...
copy /y aqi_model_fixed.pkl aqi_model.pkl
copy /y aqi_features_fixed.pkl aqi_features.pkl
copy /y station_encoder_fixed.pkl station_encoder.pkl

echo Cleaning up...
del aqi_model_fixed.pkl aqi_features_fixed.pkl station_encoder_fixed.pkl
del fix_models.py

echo Pushing to GitHub...
git add aqi_model.pkl aqi_features.pkl station_encoder.pkl
git commit -m "Fix: Reload models with numpy 1.23.5 compatibility"
git push origin main

echo ✅ Done!
pause
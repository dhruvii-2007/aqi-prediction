import pickle
import numpy as np

# Try to load as raw bytes and fix
with open('aqi_model.pkl', 'rb') as f:
    data = f.read()

# Replace any numpy._core references with numpy.core
fixed_data = data.replace(b'numpy._core', b'numpy.core')

# Save fixed version
with open('aqi_model_fixed.pkl', 'wb') as f:
    f.write(fixed_data)

print("✅ Applied binary fix to model file")

# Now try to load it
try:
    with open('aqi_model_fixed.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Successfully loaded fixed model!")
    
    # Save properly with pickle
    with open('aqi_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    print("✅ Model resaved properly")
    
except Exception as e:
    print(f"❌ Still failing: {e}")

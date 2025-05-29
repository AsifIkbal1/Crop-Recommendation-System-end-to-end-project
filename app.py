import streamlit as st
import pandas as pd
import pickle
import os

# 🎯 Page configuration
st.set_page_config(page_title="🌾 Crop Recommendation", layout="centered")

# 🎉 App Title
st.title("🌾 Crop Recommendation System")
st.markdown("🔍 Find out which crop is most suitable based on Soil and Climate conditions!")

# 📥 Sidebar Inputs
st.sidebar.header("🧪 Soil & Climate Inputs")
Nitrogen = st.sidebar.slider("🌱 Nitrogen", 0, 140, 70)
Phosphorus = st.sidebar.slider("🔬 Phosphorus", 5, 150, 50)
Potassium = st.sidebar.slider("🌾 Potassium", 5, 205, 80)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", 10, 45, 25)
humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 60)
ph = st.sidebar.slider("🧪 pH", 3.5, 9.5, 6.5)
rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 0.0, 300.0, 100.0)

# 🧮 Input dataframe
input_df = pd.DataFrame({
    'Nitrogen': [Nitrogen],
    'Phosphorus': [Phosphorus],
    'Potassium': [Potassium],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})

# 📊 Show input
st.subheader("📊 Your Input Values:")
st.dataframe(input_df)

# ✅ Manual label mapping
label_map = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee',
    6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
    12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange',
    17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# 🔍 Prediction Section
try:
    if not all(os.path.exists(f"model/{f}") for f in ["random_forest_model.pkl", "scaler.pkl"]):
        raise FileNotFoundError("❌ Model or scaler file not found.")

    # Load model and scaler
    with open("model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction_encoded = model.predict(input_scaled)[0]
    prediction_label = label_map.get(prediction_encoded, "❌ Unknown crop")
    confidence = model.predict_proba(input_scaled).max() * 100

    # 🪴 Show prediction
    st.subheader("🌾 Recommended Crop:")

    # Add emojis to crop names
    crop_emojis = {
        'apple': '🍎', 'banana': '🍌', 'blackgram': '🫘', 'chickpea': '🥣', 'coconut': '🥥',
        'coffee': '☕', 'cotton': '🧵', 'grapes': '🍇', 'jute': '🧶', 'kidneybeans': '🫘',
        'lentil': '🍲', 'maize': '🌽', 'mango': '🥭', 'mothbeans': '🌱', 'mungbean': '🌿',
        'muskmelon': '🍈', 'orange': '🍊', 'papaya': '🍈', 'pigeonpeas': '🌾',
        'pomegranate': '🍎', 'rice': '🍚', 'watermelon': '🍉'
    }

    emoji = crop_emojis.get(prediction_label, "🌿")
    st.success(f" {emoji} {prediction_label.capitalize()}")
    st.markdown(f"🔢 Model Confidence: **{confidence:.2f}%**")

except FileNotFoundError as fnf_error:
    st.error(str(fnf_error))

except Exception as e:
    st.error("⚠️ An unexpected error occurred. Details below:")
    st.code(str(e))

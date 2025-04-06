import streamlit as st
import pandas as pd
import numpy as np
import joblib
import scipy.fftpack

# Load Trained Model
model = joblib.load("sleep_apnea_model.pkl")

# Feature Extraction Functions
def extract_time_features(ecg_signal):
    return {
        "mean": np.mean(ecg_signal),
        "std_dev": np.std(ecg_signal),
        "rms": np.sqrt(np.mean(np.square(ecg_signal)))
    }

def extract_frequency_features(ecg_signal):
    fft_values = scipy.fftpack.fft(ecg_signal)
    power = np.abs(fft_values) ** 2  # Power spectrum
    return {
        "low_freq_power": np.sum(power[:500]),  # Low-frequency band
        "high_freq_power": np.sum(power[500:])  # High-frequency band
    }

# Streamlit App UI
st.title("🩺 Sleep Apnea Risk Prediction")
st.write("Upload an ECG dataset to analyze the risk of sleep apnea.")

# File Upload Section
uploaded_file = st.file_uploader("Upload your ECG data (CSV)", type=["csv"])

if uploaded_file:
    st.write("📊 File successfully uploaded!")

    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.write("📄 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Extract Features
    feature_data = []
    for i in range(len(df)):
        time_features = extract_time_features(df.iloc[i, :].values)
        freq_features = extract_frequency_features(df.iloc[i, :].values)
        combined_features = {**time_features, **freq_features}
        feature_data.append(combined_features)

    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_data)

    # Make Predictions
    risk_prob = model.predict_proba(feature_df)[:, 1] * 100
    df["Risk Score (%)"] = risk_prob
    df["Risk Category"] = df["Risk Score (%)"].apply(
        lambda x: "Low" if x < 30 else "Moderate" if x < 70 else "High"
    )

    # Display Results
    st.write("📌 Sleep Apnea Risk Prediction Results:")
    st.dataframe(df[["Risk Score (%)", "Risk Category"]])

    # Show Risk Distribution
    st.bar_chart(df["Risk Score (%)"])

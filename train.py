# train_model.py - Load Data, Train ML Model, and Save Model

import pandas as pd
import joblib
import numpy as np
import scipy.fftpack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1️⃣ Load Dataset
file_path = "dataset.csv"  # Ensure the dataset is in the same folder
df = pd.read_csv(file_path)

# 2️⃣ Convert "Target" Column to Binary (0 = Normal, 1 = Sleep Apnea)
df["Target"] = df["Target"].map({"Normal": 0, "Sleep Apnea": 1})

# 3️⃣ Handle Missing Values (Replace NaN with Median)
df.fillna(df.median(), inplace=True)

# 4️⃣ Remove Outliers Using Interquartile Range (IQR)
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_clean

df_cleaned = remove_outliers(df.drop(columns=["Target"]))
df_cleaned["Target"] = df["Target"]  # Add target column back

# 5️⃣ Feature Extraction Functions

# Time-Domain Features
def extract_time_features(ecg_signal):
    return {
        "mean": np.mean(ecg_signal),
        "std_dev": np.std(ecg_signal),
        "rms": np.sqrt(np.mean(np.square(ecg_signal)))
    }

# Frequency-Domain Features (FFT)
def extract_frequency_features(ecg_signal):
    fft_values = scipy.fftpack.fft(ecg_signal)
    power = np.abs(fft_values) ** 2  # Power spectrum
    return {
        "low_freq_power": np.sum(power[:500]),  # Low-frequency band
        "high_freq_power": np.sum(power[500:])  # High-frequency band
    }

# 6️⃣ Apply Feature Extraction to All Samples
feature_data = []
for i in range(len(df_cleaned)):
    time_features = extract_time_features(df_cleaned.iloc[i, :-1].values)
    freq_features = extract_frequency_features(df_cleaned.iloc[i, :-1].values)
    combined_features = {**time_features, **freq_features}
    feature_data.append(combined_features)

# Convert to DataFrame
feature_df = pd.DataFrame(feature_data)
y = df_cleaned["Target"]

# 7️⃣ Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.2, random_state=42)

# 8️⃣ Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9️⃣ Evaluate Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 🔟 Save the Trained Model
joblib.dump(model, "sleep_apnea_model.pkl")
print("Model saved as sleep_apnea_model.pkl")

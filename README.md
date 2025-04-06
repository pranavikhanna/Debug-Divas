# 🩺 Sleep Apnea Risk Prediction App

This project is a Streamlit-based web application that predicts the risk of **Sleep Apnea** using uploaded **ECG signal data**. The app uses **machine learning models** trained on a public dataset to output a **risk score (0–100%)** and classify results into **Low, Moderate, or High risk** categories.

---

## 🚀 Demo
 
📽️ Demo Video: https://drive.google.com/file/d/19bmXA7VDHiwbwR4ctmBKQLPc4poLv4f3/view?usp=sharing

---

Project Structure

sleep_apnea_project/ ├── train.py # Script to preprocess data, extract features, train & save model ├── streamlit_app.py # Streamlit UI for file upload & prediction ├── dataset.csv # Original ECG dataset from Kaggle ├── sample_ecg_data.csv # Mini ECG sample file for testing ├── sleep_apnea_model.pkl # Trained model (Random Forest or SVM) └── requirements.txt # Python dependencies
---

Model Overview

- **Machine Learning Models Used:**
  - 🎯 Random Forest Classifier
  - 🧠 Support Vector Machine (SVM) with probability calibration
- **Training Data:** 2,500-dimensional ECG signal samples
- **Model Accuracy:**
  - Random Forest: ~98%
  - SVM: ~96%

---

Feature Engineering

To avoid overfitting and improve interpretability, we extract just **5 key features per sample** from 2,500 raw ECG points:

Time-Domain Features
- **Mean** — Average signal amplitude  
- **Standard Deviation** — Variability in ECG waveform  
- **RMS (Root Mean Square)** — Signal energy

Frequency-Domain Features (via FFT)
- **Low Frequency Power** — Captures low band variation (first 500 coefficients)  
- **High Frequency Power** — Captures upper signal spectrum (rest of FFT)

---

## 🖥️ How the App Works

1. Upload a CSV containing raw ECG data (rows of numeric values).
2. The app automatically extracts features from each row.
3. The model predicts a **risk score (0–100%)**.
4. Each row is labeled with a **risk category**:
   - 🟢 Low Risk: <30%
   - 🟡 Moderate Risk: 30–70%
   - 🔴 High Risk: >70%
5. A **bar chart** visualizes risk scores across samples.

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections.abc import Sequence  # Python 3.13 compatibility

# Sample Data: Simulating a time-series physiological signal (e.g., ECG, respiratory)
def generate_sample_data(size: int = 1000, fs: int = 100) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, size / fs, size)
    ecg_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, size)
    return t, ecg_signal

# Feature Extraction Functions

def time_domain_features(signal_data: np.ndarray) -> dict[str, float]:
    features = {
        "Mean": float(np.mean(signal_data)),
        "Median": float(np.median(signal_data)),
        "Variance": float(np.var(signal_data)),
        "Peak-to-Peak": float(np.ptp(signal_data)),
        "RMS": float(np.sqrt(np.mean(signal_data**2))),
        "Skewness": float(stats.skew(signal_data)),
        "Kurtosis": float(stats.kurtosis(signal_data)),
        "Entropy": float(stats.entropy(np.histogram(signal_data, bins=10)[0] + 1e-6))
    }
    return features


def frequency_domain_features(signal_data: np.ndarray, fs: int = 100) -> dict[str, float]:
    freqs, psd = signal.welch(signal_data, fs)
    features = {
        "Power Spectral Density Mean": float(np.mean(psd)),
        "Low-Frequency Power": float(np.sum(psd[(freqs >= 0.1) & (freqs < 0.5)])),
        "High-Frequency Power": float(np.sum(psd[(freqs >= 0.5) & (freqs < 2.5)])),
        "Spectral Entropy": float(stats.entropy(psd + 1e-6))
    }
    return features


def statistical_ml_features(signal_data: np.ndarray, n_components: int = 3) -> dict[str, float]:
    # Principal Component Analysis (PCA)
    pca = PCA(n_components=n_components)
    reshaped_data = signal_data.reshape(-1, 1)
    pca_features = pca.fit_transform(reshaped_data)
    
    # t-SNE Embedding
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(reshaped_data)
    
    features = {
        "PCA Component 1": float(np.mean(pca_features[:, 0])),
        "PCA Component 2": float(np.mean(pca_features[:, 1])) if n_components > 1 else 0.0,
        "t-SNE Dim 1": float(np.mean(tsne_features[:, 0])),
        "t-SNE Dim 2": float(np.mean(tsne_features[:, 1]))
    }
    return features


if __name__ == "__main__":
    # Generate Sample Data
    t, ecg_signal = generate_sample_data()

    # Extract Features
    time_features = time_domain_features(ecg_signal)
    frequency_features = frequency_domain_features(ecg_signal)
    ml_features = statistical_ml_features(ecg_signal)

    # Print Extracted Features
    print("Time-Domain Features:", time_features)
    print("Frequency-Domain Features:", frequency_features)
    print("Statistical & ML-Based Features:", ml_features)

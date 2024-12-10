import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract relevant speech features for classification.

    Parameters:
        file_path (str): Path to the audio file.

    Returns:
        features (np.ndarray): Array of extracted features.
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio file.
    """
    y, sr = librosa.load(file_path)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  # MFCCs
    jitter = np.std(librosa.feature.zero_crossing_rate(y).T)  # Scalar
    shimmer = np.std(librosa.feature.rms(y=y).T)  # Scalar
    fundamental_freq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # Scalar

    # Combine features into a single 1D array
    features = np.hstack([mfccs, jitter, shimmer, fundamental_freq])
    return features, y, sr

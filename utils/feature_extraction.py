import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.stats import variation
import librosa
import parselmouth  # For formant analysis

def compute_mfccs(y, sr, n_mfcc=13):
    """Compute custom MFCCs."""
    n_fft = 2048
    hop_length = 512
    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft)  # Use keyword arguments
    mel_spectrum = np.dot(mel_filters, spectrum)
    log_mel_spectrum = np.log(mel_spectrum + 1e-9)
    mfccs = fftpack.dct(log_mel_spectrum, axis=0, type=2, norm='ortho')[:n_mfcc]
    return np.mean(mfccs, axis=1)

def compute_jitter(y, sr):
    """Compute jitter (variation in pitch)."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    nonzero_pitches = pitches[pitches > 0]
    if len(nonzero_pitches) > 1:
        jitter = np.mean(np.abs(np.diff(nonzero_pitches)) / nonzero_pitches[:-1])
    else:
        jitter = 0
    return jitter

def compute_shimmer(y, sr):
    """Compute shimmer (variation in amplitude)."""
    frame_size = int(0.03 * sr)  # 30ms
    hop_size = int(0.01 * sr)    # 10ms
    frames = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_size)
    amplitudes = np.max(np.abs(frames), axis=0)
    if len(amplitudes) > 1:
        shimmer = np.std(np.diff(amplitudes)) / np.mean(amplitudes)
    else:
        shimmer = 0
    return shimmer

def compute_fundamental_freq(y, sr):
    """Compute fundamental frequency (pitch F0)."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    nonzero_pitches = pitches[pitches > 0]
    fundamental_freq = np.mean(nonzero_pitches) if len(nonzero_pitches) > 0 else 0
    return fundamental_freq

def compute_formant_frequencies(file_path):
    """Compute formant frequencies using Praat."""
    sound = parselmouth.Sound(file_path)
    formant = sound.to_formant_burg()
    formants = []
    for t in np.linspace(0, sound.duration, 100):  # Analyze 100 time points
        f1 = formant.get_value_at_time(1, t)  # First formant (F1)
        f2 = formant.get_value_at_time(2, t)  # Second formant (F2)
        f3 = formant.get_value_at_time(3, t)  # Third formant (F3)
        formants.append((f1, f2, f3))
    formants = np.array(formants)
    formants_mean = np.nanmean(formants, axis=0)
    return formants_mean  # Return average F1, F2, F3

def compute_hnr(y, sr):
    """Compute Harmonics-to-Noise Ratio (HNR)."""
    n_fft = 2048
    hop_length = 512
    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    harmonic_energy = np.sum(spectrum[:n_fft // 2])
    noise_energy = np.sum(spectrum[n_fft // 2:])
    hnr = 10 * np.log10(harmonic_energy / noise_energy) if noise_energy > 0 else 0
    return hnr

def extract_features(file_path):
    """
    Extract custom speech features for classification.

    Parameters:
        file_path (str): Path to the audio file.

    Returns:
        features (np.ndarray): Array of extracted features.
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio file.
    """
    # Load audio
    y, sr = librosa.load(file_path)

    # Extract features using custom functions
    mfccs = compute_mfccs(y, sr)
    jitter = compute_jitter(y, sr)
    shimmer = compute_shimmer(y, sr)
    fundamental_freq = compute_fundamental_freq(y, sr)
    formant_frequencies = compute_formant_frequencies(file_path)
    hnr = compute_hnr(y, sr)

    # Combine features into a single 1D array
    features = np.hstack([mfccs, jitter, shimmer, fundamental_freq, formant_frequencies, hnr])
    return features, y, sr

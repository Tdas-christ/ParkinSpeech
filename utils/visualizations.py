import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_waveform(y, sr, file_name):
    """Plot and save waveform."""
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y)) / sr, y)
    plt.title('Speech Signal Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    waveform_path = f"./static/{file_name}_waveform.png"
    plt.savefig(waveform_path)
    plt.close()
    return waveform_path

def plot_spectrogram(y, sr, file_name):
    """Plot and save spectrogram."""
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    spectrogram_path = f"./static/{file_name}_spectrogram.png"
    plt.savefig(spectrogram_path)
    plt.close()
    return spectrogram_path

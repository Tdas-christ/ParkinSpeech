import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import parselmouth
import matplotlib.ticker as ticker

def plot_waveform(y, sr, file_name):
    """Plot and save waveform."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
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
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = f"./static/{file_name}_spectrogram.png"
    plt.savefig(spectrogram_path)
    plt.close()
    return spectrogram_path

def plot_formants(file_path, file_name):
    """Plot and save formant frequencies over time."""
    # Load audio using parselmouth
    snd = parselmouth.Sound(file_path)
    formant = snd.to_formant_burg(time_step=0.01)

    # Extract formant frequencies
    durations = np.arange(0, snd.duration, 0.01)
    f1 = [formant.get_value_at_time(1, t) for t in durations]
    f2 = [formant.get_value_at_time(2, t) for t in durations]
    f3 = [formant.get_value_at_time(3, t) for t in durations]

    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(durations, f1, label='F1', color='r')
    plt.plot(durations, f2, label='F2', color='g')
    plt.plot(durations, f3, label='F3', color='b')
    plt.title('Formant Frequencies Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.tight_layout()
    formant_path = f"./static/{file_name}_formants.png"
    plt.savefig(formant_path)
    plt.close()
    return formant_path

def plot_zero_crossing_rate(y, sr, file_name):
    """Plot and save the Zero-Crossing Rate over time."""
    frame_length = 2048
    hop_length = 512
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(zcr, sr=sr, hop_length=hop_length)

    plt.figure(figsize=(10, 4))
    plt.plot(times, zcr, label="Zero-Crossing Rate")
    plt.title("Zero-Crossing Rate Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate")
    plt.legend()
    plt.tight_layout()
    zcr_path = f"./static/{file_name}_zcr.png"
    plt.savefig(zcr_path)
    plt.close()
    return zcr_path

def plot_pitch_contour(y, sr, file_name):
    """Plot and save pitch contour."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    times = librosa.times_like(pitches, sr=sr)

    pitch_values = np.max(pitches, axis=0)
    pitch_values[pitch_values == 0] = np.nan  # Mask zeros

    plt.figure(figsize=(10, 4))
    plt.plot(times, pitch_values, label="Pitch Contour (F0)", color="g")
    plt.title("Pitch Contour Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.tight_layout()
    pitch_path = f"./static/{file_name}_pitch.png"
    plt.savefig(pitch_path)
    plt.close()
    return pitch_path



from flask import Flask, request, render_template
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load ML model
with open('./models/classification_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

def extract_features(file_path):
    """Extract detailed speech features for classification and visualization."""
    y, sr = librosa.load(file_path)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  # MFCCs
    jitter = np.std(librosa.feature.zero_crossing_rate(y).T)  # Scalar
    shimmer = np.std(librosa.feature.rms(y=y).T)  # Scalar
    fundamental_freq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # Scalar

    # Combine features into a single 1D array
    features = np.hstack([mfccs, jitter, shimmer, fundamental_freq])
    return features, y, sr

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)

        # Temporarily save the file to process it with Librosa
        file_path = f'./{filename}'
        file.save(file_path)

        # Extract features
        features, y, sr = extract_features(file_path)

        # Predict using ML model
        feature_vector = np.array(features).reshape(1, -1)
        prediction = classifier.predict(feature_vector)[0]
        class_label = "Parkinson's" if prediction == 1 else "Healthy"

        # Plot waveform
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(y)) / sr, y)
        plt.title('Speech Signal Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plot_path = f'./static/{filename}.png'
        plt.savefig(plot_path)
        plt.close()

        # Remove the temporary file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

        return render_template(
            'result.html',
            class_label=class_label,
            features=features.tolist(),
            plot_path=plot_path,
        )

if __name__ == '__main__':
    os.makedirs('./static', exist_ok=True)
    app.run(debug=True)

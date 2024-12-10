import gradio as gr
import os
import numpy as np
import librosa
from utils.feature_extraction import extract_features
from utils.visualizations import plot_waveform, plot_spectrogram
import pickle

# Load the trained model
with open('./models/classification_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

def classify_audio(file_path):
    """
    Classify the uploaded audio file and generate outputs.

    Parameters:
        file_path (str): Path to the audio file.

    Returns:
        prediction (str): Classification result.
        confidence (str): Confidence percentage.
        waveform_path (str): Path to the waveform image.
        spectrogram_path (str): Path to the spectrogram image.
        features (list): Extracted features.
    """
    try:
        # Extract features
        features, y, sr = extract_features(file_path)
        feature_vector = np.array(features).reshape(1, -1)

        # Predict using the trained model
        probabilities = classifier.predict_proba(feature_vector)[0]
        confidence = max(probabilities) * 100
        prediction = "Parkinson's" if classifier.predict(feature_vector)[0] == 1 else "Healthy"

        # File name for visualizations
        file_name = os.path.basename(file_path).split('.')[0]

        # Generate visualizations
        waveform_path = plot_waveform(y, sr, file_name)
        spectrogram_path = plot_spectrogram(y, sr, file_name)

        # Return all outputs
        return prediction, f"{confidence:.2f}%", waveform_path, spectrogram_path, features.tolist()

    except Exception as e:
        return str(e), "Error", None, None, None

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# ParkinSpeech")
    gr.Markdown("### Upload a speech file to analyze whether it belongs to a Healthy or Parkinson's patient.")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Speech File")
        analyze_button = gr.Button("Analyze")
    with gr.Row():
        prediction_output = gr.Textbox(label="Prediction")
        confidence_output = gr.Textbox(label="Confidence (%)")
    with gr.Row():
        waveform_plot = gr.Image(label="Waveform")
        spectrogram_plot = gr.Image(label="Spectrogram")
    with gr.Row():
        features_output = gr.JSON(label="Extracted Features")

    analyze_button.click(
        classify_audio,
        inputs=[audio_input],
        outputs=[prediction_output, confidence_output, waveform_plot, spectrogram_plot, features_output],
    )

# Launch the Gradio app
interface.launch()

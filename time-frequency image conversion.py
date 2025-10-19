# Install necessary dependencies (uncomment and run for the first time)
# !pip install numpy pandas pywavelets matplotlib pillow torch transformers

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
import os

# Set image display parameters
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False


def load_signal(file_path="first_sample.csv"):
    """Load signal data and preprocess"""
    try:
        df = pd.read_csv(file_path, header=None)
        signal = df.iloc[:, 0].dropna().values.astype(float)
        print(f"Signal loaded successfully, length: {len(signal)}")
        return signal
    except Exception as e:
        print(f"Failed to load signal: {str(e)}")
        raise


def cwt_to_image(signal, fs=1000, wavelet="morl", save_path="cwt_image.png"):
    """Convert signal to time-frequency image via Continuous Wavelet Transform (CWT)"""
    # Calculate wavelet scales (corresponding to frequency range)
    center_freq = pywt.central_frequency(wavelet)
    scales = np.arange(1, 200)
    frequencies = center_freq * fs / scales  # Convert scales to frequencies

    # Filter valid frequency range (10Hz to Nyquist frequency)
    valid_mask = (frequencies >= 10) & (frequencies <= fs / 2)
    scales = scales[valid_mask]
    frequencies = frequencies[valid_mask]

    # Perform CWT
    coefs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
    coefs_abs = np.abs(coefs)

    # Normalization
    coefs_norm = (coefs_abs - coefs_abs.min()) / (coefs_abs.max() - coefs_abs.min() + 1e-8)

    # Plot time-frequency image
    plt.figure(figsize=(8, 6))
    plt.imshow(
        coefs_norm,
        aspect="auto",
        cmap="jet",
        extent=[0, len(signal) / fs, frequencies[-1], frequencies[0]]
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("CWT Time-Frequency Representation")
    plt.colorbar(label="Normalized Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Return PIL image object
    return Image.open(save_path).convert("RGB")


def extract_clip_features(image):
    """Extract visual feature vectors using CLIP ViT-Base"""
    # Load model and processor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Image preprocessing
    inputs = processor(images=image, return_tensors="pt")

    # Feature extraction
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract [CLS] token features (768-dimensional)
        features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    return features


# Main workflow
if __name__ == "__main__":
    # 1. Load signal
    signal = load_signal("first_sample.csv")  # Replace with actual file path

    # 2. Generate time-frequency image
    time_freq_image = cwt_to_image(signal)
    print("Time-frequency image generated successfully")

    # 3. Extract CLIP features
    visual_features = extract_clip_features(time_freq_image)
    print(f"Feature extraction completed, dimension: {visual_features.shape}")

    # 4. Save feature vector
    np.savetxt("initial_visual_features.txt", visual_features)
    print("Feature vector saved to initial_visual_features.txt")
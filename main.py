import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
import os
import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"bearing_fault_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Set image display parameters
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

def load_signal(file_path="/mnt/first_sample.csv"):
    """
    Load signal data and preprocess it.

    This function attempts to read signal data from the specified CSV file, remove null values,
    and convert the data to a numpy array of floating - point numbers.

    Args:
        file_path (str): The path of the signal data file, default is "/mnt/first_sample.csv".

    Returns:
        np.ndarray: The loaded and preprocessed signal data array.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        signal = df.iloc[:, 0].dropna().values.astype(float)
        logging.info(f"Signal loaded successfully, length: {len(signal)}")
        return signal
    except FileNotFoundError as e:
        logging.error(f"File not found error when loading signal: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error when converting signal data to float: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred when loading signal: {e}")
        raise


def cwt_to_image(signal, fs=1000, wavelet="morl", save_path="cwt_image.png"):
    """
    Convert the signal to a time - frequency image via Continuous Wavelet Transform (CWT).

    This function first calculates the wavelet scales corresponding to the frequency range,
    then performs the continuous wavelet transform, normalizes the transform results,
    and finally plots and saves the time - frequency image.

    Args:
        signal (np.ndarray): The input signal data array.
        fs (int): The sampling frequency, default is 1000 Hz.
        wavelet (str): The wavelet basis function, default is "morl" (Morlet wavelet).
        save_path (str): The save path of the time - frequency image, default is "cwt_image.png".

    Returns:
        Image: The PIL image object of the converted time - frequency image.
    """
    try:
        # Calculate wavelet scales (corresponding to frequency range)
        center_freq = pywt.central_frequency(wavelet)
        scales = np.arange(1, 200)
        frequencies = center_freq * fs / scales  # Convert scales to frequencies

        # Filter valid frequency range (10 Hz to Nyquist frequency)
        valid_mask = (frequencies >= 10) & (frequencies <= fs / 2)
        scales = scales[valid_mask]
        frequencies = frequencies[valid_mask]

        logging.info(f"CWT scales range: {scales[0]} - {scales[-1]}, corresponding frequencies: {frequencies[-1]:.1f} - {frequencies[0]:.1f} Hz")

        # Perform CWT
        coefs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)
        coefs_abs = np.abs(coefs)

        # Normalization
        coefs_norm = (coefs_abs - coefs_abs.min()) / (coefs_abs.max() - coefs_abs.min() + 1e-8)

        # Plot time - frequency image
        plt.figure(figsize=(8, 6))
        plt.imshow(
            coefs_norm,
            aspect="auto",
            cmap="jet",
            extent=[0, len(signal) / fs, frequencies[-1], frequencies[0]]
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("CWT Time - Frequency Representation")
        plt.colorbar(label="Normalized Amplitude")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Return PIL image object
        return Image.open(save_path).convert("RGB")
    except ValueError as e:
        logging.error(f"Value error during CWT or image processing: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during CWT to image conversion: {e}")
        raise


def extract_clip_features(image):
    """
    Extract visual feature vectors from the time - frequency image using CLIP ViT - Base model.

    This function loads the CLIP image processor and visual model, preprocesses the input image,
    and then extracts the [CLS] token feature vector of the image without computing gradients.

    Args:
        image (Image): The input PIL image object.

    Returns:
        np.ndarray: The extracted 768 - dimensional visual feature vector.
    """
    try:
        # Load model and processor
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        # Image preprocessing
        inputs = processor(images=image, return_tensors="pt")

        # Feature extraction
        with torch.no_grad():
            outputs = model(**inputs)
            # Extract [CLS] token features (768 - dimensional)
            features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        logging.info(f"Feature extraction completed, dimension: {features.shape}")
        return features
    except ImportError as e:
        logging.error(f"Import error when loading CLIP model or processor: {e}")
        raise
    except RuntimeError as e:
        logging.error(f"Runtime error during feature extraction: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during CLIP feature extraction: {e}")
        raise


def generate_description(signal, features):
    """
    Generate description text based on the signal and image features.

    This function generates description text that includes the length of the raw signal
    and the dimension of the extracted image features.

    Args:
        signal (np.ndarray): The raw signal data array.
        features (np.ndarray): The extracted image feature vector.

    Returns:
        str: The generated description text.
    """
    try:
        description = f"The raw signal has a length of {len(signal)}. The extracted image features have a dimension of {features.shape}."
        logging.info("Description text generated successfully.")
        return description
    except Exception as e:
        logging.error(f"An unexpected error occurred when generating description text: {e}")
        raise


def load_external_knowledge(file_path="/mnt/External knowledge base.xlsx"):
    """
    Load the external knowledge base from an Excel file.

    This function attempts to read all sheets of the Excel file,
    combines the data of all sheets into one DataFrame,
    and finally converts the combined data into a tab - separated text format,
    where null values are represented as 'nan'.

    Args:
        file_path (str): The path of the external knowledge base Excel file, default is "/mnt/External knowledge base.xlsx".

    Returns:
        str: The text representation of the loaded external knowledge base.
    """
    try:
        excel_file = pd.ExcelFile(file_path)
        # Assume all sheets contain relevant knowledge and combine all sheets
        all_data = []
        for sheet_name in excel_file.sheet_names:
            df = excel_file.parse(sheet_name)
            all_data.append(df)
        combined_data = pd.concat(all_data, ignore_index=True)
        logging.info(f"External knowledge base loaded successfully, with {len(combined_data)} rows.")
        return combined_data.to_csv(sep='\t', na_rep='nan')
    except FileNotFoundError as e:
        logging.error(f"File not found error when loading external knowledge base: {e}")
        raise
    except ValueError as e:
        logging.error(f"Value error when parsing Excel data: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred when loading external knowledge base: {e}")
        raise


def generate_diagnosis(signal, time_freq_image, description, external_knowledge):
    """
    Generate fault diagnosis, cause analysis and repair suggestions using a large language model.

    This function constructs a prompt that includes raw signal data,
    the generation process of the time - frequency image,
    description text and the external knowledge base,
    and then calls the Qwen2.5 - 7B model in Ollama to generate the corresponding diagnosis, analysis and suggestions.

    Args:
        signal (np.ndarray): The raw signal data array.
        time_freq_image (Image): The PIL image object of the time - frequency image.
        description (str): The generated description text.
        external_knowledge (str): The text representation of the loaded external knowledge base.

    Returns:
        str: The text of fault diagnosis, cause analysis and repair suggestions generated by the large language model.
    """
    try:
        prompt = f"""
        You are an expert in bearing fault diagnosis. Please provide fault diagnosis, cause analysis and repair suggestions based on the following information:
        1. Raw signal data: {signal.tolist()}
        2. Time - frequency image (represented by its generation process and saved as 'cwt_image.png')
        3. Description text: {description}
        4. External knowledge base: {external_knowledge}
        """

        url = "http://localhost:11434/api/generate"
        data = {
            "model": "qwen2.5-7b",
            "prompt": prompt
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            result = ""
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line)
                    if "response" in json_data:
                        result += json_data["response"]
            logging.info("Successfully obtained response from the language model.")
            return result
        else:
            logging.error(f"Failed to get response from the language model. Status code: {response.status_code}")
            raise Exception(f"Failed to get response from the language model. Status code: {response.status_code}")
    except requests.ConnectionError as e:
        logging.error(f"Connection error when calling the language model: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error when processing the response from the language model: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred when generating diagnosis: {e}")
        raise


if __name__ == "__main__":
    try:
        # 1. Load signal
        signal = load_signal()

        # 2. Generate time - frequency image
        time_freq_image = cwt_to_image(signal)

        # 3. Extract image features
        features = extract_clip_features(time_freq_image)

        # 4. Generate description text
        description = generate_description(signal, features)

        # 5. Load external knowledge base
        external_knowledge = load_external_knowledge()

        # 6. Generate diagnosis, cause analysis and repair suggestions
        result = generate_diagnosis(signal, time_freq_image, description, external_knowledge)
        print(result)
    except Exception as e:
        logging.error(f"An overall error occurred during the entire process: {e}")
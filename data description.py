import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# 1. Data Loading and Preprocessing
# --------------------------
def load_bearing_data(file_path="first_sample.csv"):
    """Load bearing data and process format"""
    # Read CSV data (handle single-column numeric format)
    df = pd.read_csv(file_path, header=None, names=["bearing_value"])
    # Remove null values and abnormally formatted data
    df = df.dropna()
    df["bearing_value"] = pd.to_numeric(df["bearing_value"], errors="coerce")
    df = df.dropna()
    return df["bearing_value"].values  # Return numeric array

# --------------------------
# 2. 12 Indicators Calculation Function
# --------------------------
def calculate_bearing_indicators(data):
    """
    Calculate 12 key indicators of bearing data:
    1. Mean value
    2. Standard deviation
    3. Absolute mean value
    4. Peak value
    5. Variance
    6. Waveform index
    7. Frequency mean value
    8. Frequency variance
    9. Gravity frequency
    10. Frequency standard deviation
    11. Frequency root mean square
    12. Average frequency
    """
    # Time-domain indicator calculation
    mean_val = np.mean(data)  # 1. Mean value
    std_val = np.std(data, ddof=1)  # 2. Standard deviation (sample standard deviation)
    abs_mean_val = np.mean(np.abs(data))  # 3. Absolute mean value
    peak_val = np.max(np.abs(data))  # 4. Peak value (maximum of absolute values)
    variance_val = np.var(data, ddof=1)  # 5. Variance (sample variance)
    rms_val = np.sqrt(np.mean(data **2))  # Auxiliary calculation: root mean square
    waveform_index = peak_val / rms_val if rms_val != 0 else 0  # 6. Waveform index

    # Frequency-domain indicator calculation (based on FFT)
    n = len(data)
    fs = 1000  # Assume sampling frequency 1000Hz (adjust according to actual scenario)
    fft_result = np.fft.fft(data)
    fft_mag = np.abs(fft_result)[:n // 2]  # Take magnitude of positive frequency part
    freq = np.fft.fftfreq(n, 1 / fs)[:n // 2]  # Positive frequency axis

    # Filter frequency points with zero magnitude (avoid calculation errors)
    valid_mask = fft_mag > 1e-10
    valid_freq = freq[valid_mask]
    valid_mag = fft_mag[valid_mask]
    if len(valid_freq) == 0:
        valid_freq = np.array([0])
        valid_mag = np.array([0])

    # Frequency-domain indicator calculation
    freq_mean = np.mean(valid_freq)  # 7. Frequency mean value
    freq_var = np.var(valid_freq, ddof=1)  # 8. Frequency variance
    gravity_freq = np.sum(valid_freq * valid_mag) / np.sum(valid_mag) if np.sum(valid_mag) != 0 else 0  # 9. Gravity frequency
    freq_std = np.std(valid_freq, ddof=1)  # 10. Frequency standard deviation
    freq_rms = np.sqrt(np.mean(valid_freq** 2))  # 11. Frequency root mean square
    avg_freq = np.sum(valid_freq * (valid_mag / np.sum(valid_mag))) if np.sum(valid_mag) != 0 else 0  # 12. Average frequency

    # Organize indicator results (retain 6 decimal places, unified format)
    indicators = {
        "1. Mean value": round(mean_val, 6),
        "2. Standard deviation": round(std_val, 6),
        "3. Absolute mean value": round(abs_mean_val, 6),
        "4. Peak value": round(peak_val, 6),
        "5. Variance": round(variance_val, 6),
        "6. Waveform index": round(waveform_index, 6),
        "7. Frequency mean value": round(freq_mean, 6),
        "8. Frequency variance": round(freq_var, 6),
        "9. Gravity frequency": round(gravity_freq, 6),
        "10. Frequency standard deviation": round(freq_std, 6),
        "11. Frequency root mean square": round(freq_rms, 6),
        "12. Average frequency": round(avg_freq, 6)
    }
    return indicators

# --------------------------
# 3. Qwen2.5-7B Model Loading and Description Generation
# --------------------------
def load_qwen2_5_model(model_path="qwen/Qwen2.5-7B"):
    """Load Qwen2.5-7B model and Tokenizer (need to install transformers, accelerate in advance)"""
    # Automatically load model and Tokenizer (support CPU/GPU, GPU requires CUDA environment)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 16-bit precision to save memory
        device_map="auto",  # Automatically allocate device (prioritize GPU)
        trust_remote_code=True
    )
    model.eval()  # Inference mode
    return tokenizer, model

def generate_description_with_qwen(indicators, tokenizer, model):
    """Generate natural language description based on 12 indicators"""
    # Construct Prompt (clarify task requirements, guide professional description)
    prompt = f"""
    You are an expert in bearing fault diagnosis data analysis. Please generate a structured text description based on the following 12 key indicators of bearing data:
    1. First, outline the basic situation of the data (data volume, indicator coverage);
    2. Explain the meaning of each indicator and the physical significance of the current calculation results item by item (analyze in combination with bearing operating status);
    3. The language should be professional and concise, avoiding redundancy, suitable for engineering report scenarios.

    Calculation results of 12 indicators of bearing data:
    {chr(10).join([f"{k}: {v}" for k, v in indicators.items()])}

    Please generate text description:
    """

    # Model inference (set generation parameters)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,  # Maximum number of generated words
            temperature=0.7,  # Randomness (0.7 is suitable for professional description)
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and organize results
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract model-generated part (remove original Prompt)
    description = description.split("Please generate text description:")[-1].strip()
    return description

# --------------------------
# 4. Main Execution Process
# --------------------------
if __name__ == "__main__":
    # Step 1: Load bearing data
    print("Loading bearing data...")
    bearing_data = load_bearing_data("first_sample.csv")
    print(f"Data loading completed, total {len(bearing_data)} sampling points\n")

    # Step 2: Calculate 12 indicators
    print("Calculating 12 bearing indicators...")
    indicators = calculate_bearing_indicators(bearing_data)
    # Print indicator results
    print("12 indicators calculation results:")
    for k, v in indicators.items():
        print(f"  {k}: {v}")
    print()

    # Step 3: Load Qwen2.5-7B model
    print("Loading Qwen2.5-7B model... (Model download is required for first run, which may take a long time)")
    tokenizer, model = load_qwen2_5_model()
    print("Model loading completed\n")

    # Step 4: Generate text description
    print("Generating bearing data text description...")
    description = generate_description_with_qwen(indicators, tokenizer, model)

    # Step 5: Save results
    with open("bearing_indicators_description.txt", "w", encoding="utf-8") as f:
        f.write("Bearing data 12 indicators calculation results:\n")
        f.write(chr(10).join([f"{k}: {v}" for k, v in indicators.items()]))
        f.write("\n\n=== Model-generated text description ===\n")
        f.write(description)

    # Print final results
    print("\n=== Model-generated bearing data text description ===")
    print(description)
    print(f"\nResults saved to: bearing_indicators_description.txt")
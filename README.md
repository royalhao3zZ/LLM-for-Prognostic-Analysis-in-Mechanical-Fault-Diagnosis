Bearing Fault Diagnosis System: An Intelligent Analysis Tool Based on Multimodal Data and Large Language Models
Project Overview
This repository contains a comprehensive bearing fault diagnosis system that integrates signal processing, time-frequency analysis, computer vision, and large language models (LLMs) to provide accurate fault diagnosis, cause analysis, and maintenance recommendations. The system processes raw bearing vibration signals, converts them into time-frequency images via Continuous Wavelet Transform (CWT), extracts visual features using the CLIP ViT-Base model, and leverages the Qwen2.5-7B model (via Ollama) to generate professional diagnostic reports combined with external knowledge bases.
Core Features
Signal Loading and Preprocessing: Read raw bearing vibration signals and perform cleaning
Time-Frequency Analysis: Convert 1D signals to 2D time-frequency images using Continuous Wavelet Transform (CWT)
Feature Extraction: Extract visual features from time-frequency images using the CLIP ViT-Base model
Multimodal Data Fusion: Integrate raw signals, time-frequency image features, descriptive text, and external knowledge bases
Intelligent Diagnosis: Call the Qwen2.5-7B model to generate fault diagnosis results, cause analysis, and maintenance recommendations
Environment Requirements
Required Dependencies
Python 3.8+
Data processing libraries: numpy, pandas
Signal processing library: pywt (PyWavelets)
Image processing libraries: matplotlib, Pillow
Machine learning framework: torch (PyTorch)
Transformer model library: transformers
API communication library: requests
Excel processing library: openpyxl (for external knowledge base)
Model Dependencies
CLIP ViT-Base model (automatically downloaded)
Ollama service + Qwen2.5-7B model
Installation reference: Ollama Official Documentation
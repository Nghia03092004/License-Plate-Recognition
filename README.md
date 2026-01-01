# 🚗 Vietnamese License Plate Recognition System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green?style=for-the-badge&logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://license-plate-recognition-demo.streamlit.app)

A high-performance Computer Vision system designed to detect and recognize Vietnamese license plates (1-line and 2-line formats) in real-time. This project combines **YOLOv8** for robust object detection with a customized **Coordinate Sorting Algorithm** and **EasyOCR** for accurate character recognition.

The system has been heavily optimized for CPU performance (Intel Core i7), achieving a **2.5x speed increase** compared to the baseline implementation.

> **🔴 LIVE DEMO:** [Click here to try the App](https://license-plate-recognition-demo.streamlit.app)

## 🚀 Key Features

* **Robust Detection:** Utilizes **YOLOv8** (State-of-the-Art) to detect license plates in various conditions (slanted angles, low light, complex backgrounds).
* **Optimized Performance:** Runs efficiently on CPU-only hardware (Intel i7) with low latency (~200ms/image).
* **Advanced Logic:** Replaces standard OCR grouping with a custom **Coordinate Sorting Algorithm** to handle 2-line plates accurately without performance penalty.
* **Smart Preprocessing:** Implements "Smart Resizing" (only upscale when necessary) to balance between detail retention and processing speed.
* **Interactive UI:** User-friendly dashboard built with **Streamlit**.

## 📊 Performance Benchmark

The system was benchmarked on a validation set of **116 images (Kaggle Dataset)** running on an **Intel Core i7 CPU**.

| Metric | Baseline | **Optimized (Current)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Average Latency** | ~519 ms/image | **~206 ms/image** | **Reduced by 60%** |
| **Inference Speed** | 1.93 FPS | **4.85 FPS** | **2.5x Faster** |
| **Accuracy** | High | **High** | Maintained |

*Note: Benchmarking logs generated via `evaluate_project.py`.*

## 📂 Project Structure

```text
License-Plate-Project/
├─ data/                    
│  └─ *.jpg/*.png           # Test images (Kaggle Dataset)
├─ weights/                 
│  └─ plate_model.pt        # YOLOv8 weights fine-tuned for license plates
├─ venv/                    
├─ app.py                   # Streamlit Frontend
├─ core.py                  # Backend Logic (Image Processing & Inference Class)
├─ evaluate_project.py      # Script to benchmark FPS & Latency
├─ report_accuracy.csv      # Output logs from evaluation
├─ requirements.txt        
└── README.md         
```

## 🛠️ Tech Stack & Techniques

### 1. Object Detection (The "Eyes")
* **Model:** YOLOv8 (You Only Look Once - version 8).
* **Role:** Identifies the bounding box of the license plate within the full frame.

### 2. Image Processing Pipeline (The "Brain")
Before feeding the license plate into the OCR engine, the image undergoes a strict mathematical transformation in `core.py`:

1.  **Bicubic Upscaling:**
    * Resizes the plate region (x2 scale) using bicubic interpolation to enhance low-resolution details.
2.  **Grayscale Conversion:**
    * Reduces dimensionality (RGB -> Gray) to focus on pixel intensity.
3.  **Gaussian Blur:**
    * Removes high-frequency noise (grain) to prevent OCR misinterpretation.
4.  **Adaptive Thresholding:**
    * Converts the image to binary (Black/White) based on local pixel neighborhoods, making the system robust against uneven lighting and shadows.

### 3. Optical Character Recognition (OCR)
* **Library:** EasyOCR.
* **Architecture:** CRNN (Convolutional Recurrent Neural Network).
* **Post-processing:** Implements heuristic logic to clean text, filter noise, and validate the standard Vietnamese license plate format.

## 📖 Installation & Usage
1. Clone the repository
```bash
git clone <your-repo-url>
cd License-Plate-Project
```

2. Set up Virtual Environment

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Setup Data & Weights
Ensure your YOLOv8 weights file is located at weights/plate_model.pt.

(Optional) Place test images in the data/ folder.

5. Run the Application
```bash
streamlit run app.py
```
The application will launch automatically in your web browser at http://localhost:8501.

## 📝 Usage Guide
* Select Source: Choose between "Sample Data (Kaggle)" to use pre-loaded images or "Upload Image" to test with your own files.

* Run Detection: Click the primary button to trigger the pipeline.

* View Results: The system will display the bounding box on the image and print the recognized character string.

## 📝 Benchmarking
To reproduce the performance metrics on your local machine:
```bash
python evaluate_project.py
```

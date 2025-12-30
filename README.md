# 🚗 Vietnamese License Plate Recognition System

A high-performance Computer Vision system designed to detect and recognize Vietnamese license plates (1-line and 2-line formats) in real-time. This project combines **YOLOv8** for robust object detection with a custom **Image Processing Pipeline** and **EasyOCR** for accurate character recognition, accessible via an interactive Streamlit web interface.

![Demo](https://via.placeholder.com/800x400?text=Place+Your+Demo+Screenshot+Here)
*(Note: Please replace the link above with a screenshot of your actual running app)*

## 🚀 Key Features

* **Robust Detection:** Utilizes **YOLOv8** (State-of-the-Art) to detect license plates in various conditions (slanted angles, low light, complex backgrounds).
* **Advanced Image Processing:** Implements a custom matrix-manipulation pipeline (Bicubic Upscaling, Gaussian Blur, Adaptive Thresholding) to clean input data before recognition.
* **Smart OCR Engine:** Powered by **EasyOCR (CRNN architecture)** optimized for alphanumeric character extraction.
* **Interactive UI:** User-friendly dashboard built with **Streamlit**, supporting both image upload and sample dataset testing.
* **End-to-End Pipeline:** Seamless integration from raw input -> detection -> processing -> recognition -> visualization.

## 📂 Project Structure

```text
License-Plate-Project/
├─ data/                   
│  └─ *.jpg/*.png          # Test images
├─ weights/                # Model artifacts
│  └─ plate_model.pt       # YOLOv8 weights fine-tuned for license plates
├─ venv/                   
├─ app.py                  
├─ core.py                 # Backend Logic (Image Processing & Inference Class)
├─ requirements.txt       
└── README.md            
```

## 🛠️ Tech Stack & Techniques
1. Object Detection (The "Eyes")
Model: YOLOv8 (You Only Look Once - version 8).

Why: Chosen for its real-time speed and high accuracy (mAP) compared to traditional Haar Cascades or older YOLO versions.

2. Image Processing Pipeline (The "Brain")
Before feeding the license plate into the OCR engine, the image undergoes a strict mathematical transformation in core.py:

Bicubic Upscaling: Resizing the plate region (x2 scale) using bicubic interpolation to enhance low-resolution details.

Grayscale Conversion: Reducing dimensionality (RGB -> Gray) to focus on intensity.

Gaussian Blur: Removing high-frequency noise (grain) from the sensor.

Adaptive Thresholding: Converting the image to binary (Black/White) based on local pixel neighborhoods, making it robust against uneven lighting and shadows.

3. Optical Character Recognition (OCR)
Library: EasyOCR.

Architecture: CRNN (Convolutional Recurrent Neural Network).

Post-processing: Text cleaning logic to filter noise and validate license plate format.

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
Select Source: Choose between "Sample Data (Kaggle)" to use pre-loaded images or "Upload Image" to test with your own files.

Run Detection: Click the primary button to trigger the pipeline.

View Results: The system will display the bounding box on the image and print the recognized character string.

## 📄 License
This project is for educational purposes as part of an AI/ML Engineering Portfolio.
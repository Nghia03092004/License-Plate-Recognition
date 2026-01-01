import time
import cv2
import glob
import os
import pandas as pd
from ultralytics import YOLO
import easyocr
import numpy as np

# ================= CONFIGURATION =================
MODEL_PATH = 'weights/plate_model.pt' # Path to your trained YOLO model
IMAGE_FOLDER = 'data/'                # Folder containing test images
CONF_THRESHOLD = 0.25                 # Minimum confidence threshold
# =================================================

def calculate_metrics():
    print(f"Initializing system...")
    
    # 1. Load Models
    try:
        detect_model = YOLO(MODEL_PATH)
        print(f"Loaded YOLO model from {MODEL_PATH}")
    except:
        print("Custom model not found, using default 'yolov8n.pt'...")
        detect_model = YOLO('yolov8n.pt')
        
    # Set gpu=True if you have NVIDIA card, else False
    reader = easyocr.Reader(['en'], gpu=False, verbose=False) 

    # 2. Get Image List
    img_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
                glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
    
    if not img_paths:
        print(f"Error: No images found in '{IMAGE_FOLDER}'. Please add 20-50 images for testing.")
        return

    print(f"Found {len(img_paths)} images. Starting benchmark...\n")

    # List to store results
    results_log = []
    
    # Timing variables
    total_inference_time = 0
    total_frames = 0

    # --- STEP 3: TESTING LOOP ---
    for i, img_path in enumerate(img_paths):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue

        # Start Timer
        start_time = time.time()

        detections = detect_model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        
        license_text = "Not Detected"
        
        if len(detections.boxes) > 0:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop Plate
                plate_img = img[y1:y2, x1:x2]
                
                # OCR
                ocr_result = reader.readtext(plate_img, detail=0, paragraph=True)
                if ocr_result:
                    # Clean and format text
                    license_text = " - ".join(ocr_result).replace(" ", "")
                break # Only take the first plate found
        
        # End Timer
        end_time = time.time()
        
        # Calculate process time (pure inference)
        process_time = end_time - start_time
        total_inference_time += process_time
        total_frames += 1
        
        # Log data for CSV
        results_log.append({
            "Filename": filename,
            "Predicted_Text": license_text,
            "Process_Time_ms": round(process_time * 1000, 2),
            "Ground_Truth_Label": "",  
            "Is_Correct (1/0)": ""     
        })
        
        print(f"[{i+1}/{len(img_paths)}] Processed: {filename} | Time: {process_time*1000:.1f}ms | Text: {license_text}")

    # --- STEP 4: CALCULATION & REPORTING ---
    if total_frames > 0:
        avg_latency = (total_inference_time / total_frames) * 1000
        fps = 1 / (total_inference_time / total_frames)
        
        print("\n" + "="*40)
        print("SYSTEM METRICS REPORT")
        print("="*40)
        print(f"1. Total Test Images:  {total_frames}")
        print(f"2. Average Latency:    {avg_latency:.2f} ms/image")
        print(f"3. Processing Speed:   {fps:.2f} FPS")
        print("="*40)
        
        # Export CSV
        df = pd.DataFrame(results_log)
        csv_filename = "report_accuracy.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nReport saved to '{csv_filename}'.")

if __name__ == "__main__":
    calculate_metrics() 
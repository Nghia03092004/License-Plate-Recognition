import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

class LicensePlateSystem:
    def __init__(self, model_path='./weights/plate_model.pt'):
        # Load YOLO model
        try:
            self.detect_model = YOLO(model_path)
        except:
            print("Warning: License plate model not found, using default model...")
            self.detect_model = YOLO('yolov8n.pt') 
            
        self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def preprocess_plate(self, plate_img):
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        return binary

    def run(self, source_img):
        results = self.detect_model(source_img, conf=0.25, verbose=False)
        output_data = []
        annotated_img = source_img.copy()

        # Debug logs
        print(f"DEBUG: Detected {len(results[0].boxes)} objects")

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # Draw RED box (YOLO detection)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                plate_roi = source_img[y1:y2, x1:x2]
                if plate_roi.shape[0] < 10 or plate_roi.shape[1] < 10:
                    continue

                processed_roi = self.preprocess_plate(plate_roi)
                
                # OCR
                ocr_result = self.ocr_reader.readtext(processed_roi, detail=0, 
                                                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
                text_str = "".join(ocr_result).replace(" ", "").strip()
                
                print(f"DEBUG: OCR Text: {text_str}")

                if len(text_str) >= 3:
                    output_data.append(text_str)
                    # Draw GREEN box (Final Result)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_img, text_str, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return annotated_img, output_data
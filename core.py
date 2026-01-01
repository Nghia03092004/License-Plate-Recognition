import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time

# ==========================================
# PART 1: PROCESSING CLASS
# ==========================================
class LicensePlateSystem:
    def __init__(self, model_path='./weights/plate_model.pt'):
        try:
            self.detect_model = YOLO(model_path)
            print(f"Success: Loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Custom model not found, using default yolov8n.pt...")
            self.detect_model = YOLO('yolov8n.pt') 
            
        self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    def preprocess_plate(self, plate_img):
        """
        Preprocessing: Smart resize and grayscale conversion.
        """
        h, w = plate_img.shape[:2]
        
        # 1. Smart Resize: Only upscale if the plate is small (width < 200px)
        if w < 200:
            scale_factor = 2.0
            plate_img = cv2.resize(plate_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # 2. Convert to Grayscale 
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Light Blur to remove noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return blur

    def smart_ocr_parsing(self, ocr_raw_results, y_threshold=20):
        if not ocr_raw_results:
            return ""

        parsed_boxes = []
        
        for result in ocr_raw_results:
            coord, text, conf = result
            
            # Filter low confidence
            if conf < 0.2: 
                continue
                
            # Calculate box center
            y_coords = [p[1] for p in coord]
            x_coords = [p[0] for p in coord]
            cy = sum(y_coords) / 4
            cx = sum(x_coords) / 4
            
            parsed_boxes.append({"text": text, "cy": cy, "cx": cx})

        # Sort vertically (Top line first, bottom line second)
        parsed_boxes.sort(key=lambda k: k['cy'])

        if not parsed_boxes:
            return ""

        lines = []
        current_line = [parsed_boxes[0]]
        
        for i in range(1, len(parsed_boxes)):
            if abs(parsed_boxes[i]['cy'] - parsed_boxes[i-1]['cy']) < y_threshold:
                current_line.append(parsed_boxes[i])
            else:
                current_line.sort(key=lambda k: k['cx'])
                lines.append(current_line)
                current_line = [parsed_boxes[i]]
        
        current_line.sort(key=lambda k: k['cx'])
        lines.append(current_line)

        final_text = ""
        for line in lines:
            for box in line:
                final_text += box['text']
            final_text += "-"

        return final_text.replace(" ", "").strip("-")

    def run(self, source_img):
        # 1. Detect License Plate
        results = self.detect_model(source_img, conf=0.25, verbose=False)
        output_data = []
        annotated_img = source_img.copy()

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            
            for box in boxes:
                x1, y1, x2, y2 = box
                
                # Crop image
                plate_roi = source_img[y1:y2, x1:x2]
                
                # Ignore small artifacts
                if plate_roi.shape[0] < 10 or plate_roi.shape[1] < 10:
                    continue

                # 2. Preprocess (Optimization)
                processed_roi = self.preprocess_plate(plate_roi)
                
                # 3. OCR (using detail=1 for coordinates)
                ocr_result = self.ocr_reader.readtext(
                    processed_roi, 
                    detail=1, 
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
                )
                
                # 4. Smart Sorting
                text_str = self.smart_ocr_parsing(ocr_result)
                
                if len(text_str) >= 3:
                    output_data.append(text_str)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw text background and text
                    (w, h), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_img, (x1, y1 - 30), (x1 + w, y1), (0, 255, 0), -1)
                    
                    cv2.putText(annotated_img, text_str, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return annotated_img, output_data

if __name__ == "__main__":
    print("Initializing AI model...")
    lps = LicensePlateSystem()
    
    # Open Camera or Video (Change 'video.mp4' to 0 for webcam)
    cap = cv2.VideoCapture(0) 

    # Optimization
    FRAME_SKIP = 4       # Process 1 frame every 4 frames 
    RESIZE_WIDTH = 640   # Reduce input width to 640px for YOLO speed

    frame_count = 0
    last_annotated_frame = None # Store previous result
    
    print("Starting system...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera/video or stream ended.")
            break

        # 1. RESIZE INPUT
        h, w = frame.shape[:2]
        if w > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w
            dim = (RESIZE_WIDTH, int(h * scale))
            frame_processed = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        else:
            frame_processed = frame

        # 2. FRAME SKIPPING 
        if frame_count % FRAME_SKIP == 0:
            start_time = time.time()
            
            annotated, results = lps.run(frame_processed)
            
            last_annotated_frame = annotated
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            if results:
                print(f"Plate: {results} | FPS: {fps:.2f}")
        else:
            annotated = last_annotated_frame if last_annotated_frame is not None else frame_processed

        cv2.imshow("License Plate Recognition System", annotated)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
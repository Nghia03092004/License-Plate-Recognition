import streamlit as st
import cv2
import numpy as np
import os
from core import LicensePlateSystem 

# Page Configuration
st.set_page_config(page_title="License Plate Project", layout="wide")
st.title("License Plate Recognition (YOLO + OCR)")

# Cache model
@st.cache_resource
def load_system():
    return LicensePlateSystem(model_path='./weights/plate_model.pt')

try:
    system = load_system()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Configuration")
source_type = st.sidebar.radio("Image Source:", ["Sample Data (Kaggle)", "Upload Image"])

input_img = None

if source_type == "Sample Data (Kaggle)":
    data_path = "./data"
    if os.path.exists(data_path):
        files = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.png'))]
        if files:
            selected_file = st.sidebar.selectbox("Select file:", files)
            file_path = os.path.join(data_path, selected_file)
            input_img = cv2.imread(file_path)
        else:
            st.sidebar.warning("Data folder is empty!")
    else:
        st.sidebar.error("Data folder not found")

elif source_type == "Upload Image":
    uploaded = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        input_img = cv2.imdecode(file_bytes, 1)

if input_img is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("RUN DETECTION", type="primary"):
        with st.spinner("Processing..."):
            result_img, plates = system.run(input_img)
            
            with col2:
                st.subheader("Result")
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if plates:
                st.success(f"Detected Plates: {plates}")
            else:
                st.warning("No plates detected (Check CMD for red bounding boxes)")